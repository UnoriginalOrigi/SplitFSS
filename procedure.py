import cProfile
import io
import pstats
import time
import pickle
import sys
import os

import torch
import numpy as np
import syft as sy


def profile(func):
    """A gentle profiler"""

    def wrapper(args_, *args, **kwargs):
        if args_.verbose:
            pr = cProfile.Profile()
            pr.enable()
            retval = func(args_, *args, **kwargs)
            pr.disable()
            s = io.StringIO()
            ps = pstats.Stats(pr, stream=s).sort_stats("tottime")
            ps.print_stats(0.1)
            print(s.getvalue())
            return retval
        else:
            return func(args_, *args, **kwargs)

    return wrapper

#PUBLIC DATA EXTRACTION
def extract_public_data(data_loader):
    dataPub = []
    targetPub = []
    for batch_idx, (data, target) in enumerate(data_loader):
        dataPub.append(data)
        targetPub.append(target)
    return dataPub, targetPub


def train(args, kwargs, modelPub, modelPriv, private_train_loader, public_train_loader, optimizerPub, optimizerPriv, comms_count, client_server_comms, server_client_comms, epoch):
    if not modelPub == None:
        modelPub.train()
    modelPriv.train()
    times = []
    try:
        n_items = (len(public_train_loader) - 1) * args.batch_size + len(
            public_train_loader[-1][1]
        )
    except TypeError:
        n_items = len(public_train_loader.dataset)

    #EXTRACTING PUBLIC DATA FROM THE TRAIN LOADER
    dataPub, targetPub = extract_public_data(public_train_loader)

    for batch_idx, (data, target) in enumerate(private_train_loader):
        if args.comm_info:
            sy.comm_total = 0
        start_time = time.time()

        #FORWARD REQUIRES TO PASS THE PUBLIC DATA AND THE SPLIT MODELS NOW
        def forward(optimizerPub, modelPub, modelPriv, data, dataPub, target, targetPub, comms_count, client_server_comms, server_client_comms):

            if not modelPub == None:
                optimizerPub.zero_grad()    
            optimizerPriv.zero_grad()
  
            #SPLIT TRAINING PUBLIC ON CLIENT; SECRET SHARING ON SERVER
            #Creating Training output
            if not modelPub == None:
                output = modelPub(dataPub[batch_idx])

                #encrypting/sending to the servers
                if not args.public:
                    output = output.encrypt(**kwargs)
                    if args.comm_info:
                        comms_count += sy.comm_total 
                        client_server_comms += sy.comm_total
                        backup_comms = sy.comm_total

                else:
                    if args.comm_info:
                        output_comms = len(pickle.dumps(output))
                        comms_count += output_comms
                        client_server_comms += output_comms
                #operating on secret shared values
                output = modelPriv(output)
                
                if args.model == "usplit": #usplit implementation
                    if not args.public:   
                        if args.comm_info:
                            alice_shares = output.child.child.child.child['alice'].copy().get()
                            bob_shares = output.child.child.child.child['bob'].copy().get()
                            alice_comms = len(pickle.dumps(alice_shares))
                            bob_comms = len(pickle.dumps(bob_shares))
                            comms_count += alice_comms + bob_comms
                            server_client_comms += alice_comms + bob_comms                   
                        output = output.get()

                    else:
                        if args.comm_info:
                            output_comms = len(pickle.dumps(output))
                            comms_count += output_comms
                            server_client_comms += output_comms
                    final_layer = torch.nn.ReLU()
                    output = final_layer(output)

            else: #local implementation
                output = modelPriv(data)
                if args.comm_info:
                    if not args.public:                        
                        shares = data.child.child.child.child['alice'].copy().get()
                        share_comms = len(pickle.dumps(shares))
                        comms_count += share_comms
                        client_server_comms += share_comms
                    else:
                        output_comms = len(pickle.dumps(dataPub[batch_idx]))
                        comms_count += output_comms
                        client_server_comms += output_comms

            batch_size = output.shape[0]
            if args.comm_info:
                if not args.public:                        
                    alice_shares = target.child.child.child.child['alice'].copy().get()
                    bob_shares = target.child.child.child.child['bob'].copy().get()
                    alice_comms = len(pickle.dumps(alice_shares))
                    bob_comms = len(pickle.dumps(bob_shares))
                    comms_count += alice_comms + bob_comms
                    client_server_comms += alice_comms + bob_comms
                else:
                    output_comms = len(pickle.dumps(output))
                    target_comms = len(pickle.dumps(target))
                    comms_count += output_comms + target_comms
                    client_server_comms += target_comms
            #print("Calc target Comms: ", time.time() - start_time)
            # if args.model in {"lefull", "lesplit"}:
            #     if args.public:
            #         target_test = torch.argmax(target, dim=1)
            #         loss_enc = torch.nn.functional.cross_entropy(output, target_test)
            #     else:
            #         loss_enc = output.cross_entropy(target)
            # else:
            try:
                loss_enc = ((target-output)**2).sum() / batch_size
            except sy.exceptions.PureFrameworkTensorFoundError:
                loss_enc = ((targetPub[batch_idx]-output)**2).sum() / batch_size

            return loss_enc, comms_count, client_server_comms, server_client_comms

        loss = [10e10]
        loss_dec = torch.tensor([10e10])

        while loss_dec.abs() > 15:
            loss[0], comms_count, client_server_comms, server_client_comms = forward(optimizerPub, modelPub, modelPriv, data, dataPub, target, targetPub, comms_count, client_server_comms, server_client_comms)
            loss_dec = loss[0].copy()
            if loss_dec.is_wrapper:
                if not args.fp_only:
                    if args.comm_info:
                        if not args.model == "usplit":
                            alice_shares = loss_dec.child.child.child.child['alice'].copy().get()
                            bob_shares = loss_dec.child.child.child.child['bob'].copy().get()
                            loss_send = len(pickle.dumps(alice_shares)) + len(pickle.dumps(bob_shares))
                            comms_count += loss_send
                            server_client_comms += loss_send
                    loss_dec = loss_dec.get()
                loss_dec = loss_dec.float_precision()
                if args.comm_info:
                    if args.model == "usplit":
                        sy.comm_total = 0
                        loss_send = loss_dec.encrypt(**kwargs)
                        comms_count += loss_send
                        client_server_comms += sy.comm_total
                        
                #print(loss_dec)
            if loss_dec.abs() > 15:
                print(f'⚠️ #{batch_idx} loss:{loss_dec.item()} RETRY...')        
        
        loss[0].backward()

        if not optimizerPub == None:
            optimizerPub.step()
        optimizerPriv.step()
        tot_time = time.time() - start_time
        times.append(tot_time)
        if args.comm_info:
                del sy.comm_total
        if batch_idx % args.log_interval == 0:
            if args.train:
                print(
                    "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tTime: {:.3f}s ({:.3f}s/item) [{:.3f}]".format(
                        epoch,
                        batch_idx * args.batch_size,
                        n_items,
                        100.0 * batch_idx / len(private_train_loader),
                        loss_dec.item(),
                        tot_time,
                        tot_time / args.batch_size,
                        args.batch_size,
                    )
                )
    return torch.tensor(times).sum().item(), comms_count, client_server_comms, server_client_comms

#TEST IS ALSO UPDATED TO DO THE SAME. SPLIT MODELS, ALSO PRIVATE AND PUBLIC MODELS
@profile
def test(args, kwargs, modelPub, modelPriv, private_test_loader, public_test_loader):
    if not modelPub == None:
        modelPub.eval()
    modelPriv.eval()
    test_comms = 0
    correct = 0
    times = 0
    real_times = 0  # with the argmax
    i = 0
    try:
        n_items = (len(private_test_loader) - 1) * args.test_batch_size + len(
            private_test_loader[-1][1]
        )
    except TypeError:
        n_items = len(private_test_loader.dataset)

    #PUBLIC DATA EXTRACTION
    dataPub, targetPub = extract_public_data(public_test_loader)
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(private_test_loader):
            if args.comm_info:
                sy.comm_total = 0
            i += 1
            start_time = time.time()
                
            #SAME CONCEPT AS TRAIN.
            if not modelPub == None:
                output = modelPub(dataPub[batch_idx])
                if not args.public:
                    output = output.encrypt(**kwargs)                    
                if args.comm_info:
                    if not args.public:                        
                        test_comms += sy.comm_total
                    else:
                        output_comms = len(pickle.dumps(output))
                        test_comms += output_comms

                output = modelPriv(output)
                if args.model == "usplit":
                    if args.comm_info:
                        if not args.public:                        
                            output_comms = len(pickle.dumps(output))
                            test_comms += output_comms
                        else:
                            output_comms = len(pickle.dumps(output))
                            test_comms += output_comms
                    if not args.public:
                        output = output.get()
                    final_layer = torch.nn.ReLU()
                    output = final_layer(output)
            else:
                output = modelPriv(data)
                if args.comm_info:
                    if not args.public:                        
                        alice_shares = data.child.child.child.child['alice'].copy().get()
                        bob_shares = data.child.child.child.child['bob'].copy().get()
                        alice_comms = len(pickle.dumps(alice_shares))
                        bob_comms = len(pickle.dumps(bob_shares))
                        test_comms += alice_comms + bob_comms
                    else:
                        output_comms = len(pickle.dumps(data))
                        test_comms += output_comms

            times += time.time() - start_time
            pred = output.argmax(dim=1)
            real_times += time.time() - start_time

            try:
                correct += pred.eq(target.view_as(pred)).sum()
            except sy.exceptions.PureFrameworkTensorFoundError:
                target1 = target.copy().get()
                correct += pred.eq(target1.view_as(pred)).sum()
            if args.comm_info:
                del sy.comm_total
            if batch_idx % args.log_interval == 0 and correct.is_wrapper:
                if args.fp_only or args.model == "usplit":
                    c = correct.copy().float_precision()
                else:
                    c = correct.copy().get().float_precision()
                ni = i * args.test_batch_size
            
                if args.test:
                    print(
                        "Accuracy: {}/{} ({:.0f}%) \tTime / item: {:.4f}s".format(
                            int(c.item()),
                            ni,
                            100.0 * c.item() / ni,
                            times / ni,
                        )
                    )

    if correct.is_wrapper:
        if args.fp_only or args.model == "usplit":
            correct = correct.float_precision()
        else:
            correct = correct.get().float_precision()
    if args.comm_info:
        print(f"\nTesting communications overhead: {test_comms} bytes")
    if args.test:
        print(
            "TEST Accuracy: {}/{} ({:.2f}%) \tTime /item: {:.4f}s \tTime w. argmax /item: {:.4f}s [{:.3f}]\n".format(
                correct.item(),
                n_items,
                100.0 * correct.item() / n_items,
                times / n_items,
                real_times / n_items,
                args.test_batch_size,
            )
        )

    return times, test_comms
