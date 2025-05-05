from multiprocessing import Lock, Process, Queue, current_process
import time
import os
import queue # imported for using queue.Empty exception
import numpy as np
import utils

import warnings
warnings.simplefilter(action='ignore', category=(FutureWarning, RuntimeWarning))

import contextlib
import io
import itertools
import sys
from ml_collections import ConfigDict

def make_configs():
    #----------------------------
    seed_list = [x for x in range(1)]

    # wd_list = [0.01, 0.]
    # gn_clip_list = [None, 1.]
    # s = [seed_list, wd_list, gn_clip_list]

    stride_k_list = [1, 2, 4]
    b2_list = [0.99, 0.999]
    s = [seed_list, stride_k_list, b2_list]

    hyp_list = list(itertools.product(*s))

    sweep_configs = []
    for hyp in hyp_list:
        model_config = ConfigDict()
        optim_config = ConfigDict()
        data_config = ConfigDict()
        log_config = ConfigDict()
        cb_config = ConfigDict()

        #--------------------------------
        optim_config.seed = hyp[0]

        data_config.stride = int(1024/hyp[1])
        data_config.n_train = 2335*hyp[1]
        optim_config.b2 = hyp[2]
        optim_config.n_epochs = 60

        # optim_config.wd = hyp[1]
        # optim_config.gn_clip = hyp[2]

        cfg = ConfigDict(
            dict(
                model=model_config,
                optim=optim_config,
                data=data_config,
                log=log_config,
                cb=cb_config,
            )
        )
        sweep_configs.append(cfg)

    return sweep_configs


@contextlib.contextmanager
def nostdout():
    save_stdout = sys.stdout
    sys.stdout = open('trash', 'w')
    yield
    sys.stdout = save_stdout

def do_job(tasks_to_accomplish, tasks_that_are_done, job):
    datasets = None
    while True:
        try:
            '''
                try to get task from the queue. get_nowait() function will 
                raise queue.Empty exception if the queue is empty. 
                queue(False) function would do the same task also.
            '''
            task = tasks_to_accomplish.get_nowait()
            # print(current_process().name)
            # print(int(current_process().name[-1:]))
            process_id = int(current_process().name[-1:])
            os.environ['CUDA_VISIBLE_DEVICES'] = str(process_id-1)
            os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.9'
            # os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

            out_str, mh, datasets = job(task, datasets, resume=False)
            print(out_str)
            # print(mh)
            print("Train Perp")
            print(np.array(list(mh['train_perplexity'])))
            print("Test Perp")
            print(np.array(list(mh['test_perplexity'])))
            print("Train Acc")
            print(np.array(list(mh['train_accuracy'])))
            print("Test Acc")
            print(np.array(list(mh['test_accuracy'])))
            #
            time.sleep(0.2)

        except queue.Empty:

            break
        else:
            '''
                if no exception has been raised, add the task completion 
                message to task_that_are_done queue
            '''
            # print(task)
            out = str(task) + f' is done by {current_process().name}; test accuracy {mh["test_accuracy"][-1]:%}; lse {mh["lse"]};'
            try:
                ind = np.where(np.array(mh['train_accuracy'][:])>0.999)[0][0]
                out += f" first_99_acc: {mh['test_accuracy'][ind]:%}; cross epoch:[{ind}];"
            except IndexError:
                out += f" first_99_acc: na; cross epoch:[na];"
            tasks_that_are_done.put(out)
            time.sleep(.5)
    return True


def main():

    # import utils
    # # mh = utils.load_thing("traj/250502-1158_wiki2_2335_276_GPT2-small-pretrained_seed0_sgdFam_1b0.9_2b0.999_3b0.0_lr0.005_warmup2_wd0.005_bs8/metrics.pkl")
    # mh = utils.load_thing("traj/250502-1403_wiki2_2335_276_stride1024_GPT2-small-pretrained_seed0_sgdFam_1b0.9_2b0.999_3b0.0_lr5e-05_warmup2_wd0.005_bs8/metrics.pkl")
    #
    # print(np.array(list(mh['train_perplexity'])))
    # print(np.array(list(mh['test_perplexity'])))
    #
    # return

    from train_scripts.gpt2mini_ft_wiki2_train import train_model

    ### Flexible HPs
    configs = make_configs()

    number_of_tasks = len(configs)
    print(number_of_tasks, "tasks")
    print(configs)

    number_of_processes = 4
    process_ids = list(range(number_of_processes))
    # process_ids = [0, 1]
    # process_ids = [2, 3]

    tasks_to_accomplish = Queue()
    tasks_that_are_done = Queue()
    processes = []

    job = train_model

    for i in range(number_of_tasks):
        tasks_to_accomplish.put(configs[i])

    # creating processes
    for w in process_ids:
        p = Process(target=do_job, args=(tasks_to_accomplish, tasks_that_are_done, job), name=f"Process-N{w+1}")
        processes.append(p)
        p.start()
        time.sleep(100)  # slightly longer, was 60 should be 100

    # completing process
    for p in processes:
        p.join()

    # print the output
    while not tasks_that_are_done.empty():
        print(tasks_that_are_done.get())

    return True


if __name__ == '__main__':
    main()
