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
            os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.75'
            out_str, mh, datasets = job(task, datasets, resume=False)
            print(out_str)
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
    ### Flexible HPs
    arch_list = [
        'vgg',
        'resnet',
    ]
    sgd_hp_list = [
        (0.1, 0., 0., 0.),  # sgd
        (5e-3, 0., 0.99, 0.),  # rms
        (5e-3, 0., 0.99, -1.),  # rms-UB
        (5e-3, 0.9, 0.99, 0.),  # adam
        (5e-3, 0.9, 0.99, -1.),  # adam-UB
    ]

    # type, rho (as a scaling factor of LR), sync_period
    # sam_rho_list = [
    #     0.1,
    #     0.3,
    #     1.,
    #     3.,
    #     10.,
    # ]
    # sam_sync_list = [
    #     2,
    #     3,
    #     # 5,
    # ]
    # sam_type_list = [
    #     'sam',
    #     'asam',
    #     'lsam',
    # ]

    sam_hp_list = [
        (None, 0., 1),  # None
    ]

    # sam_hp_list += list(itertools.product(*[sam_type_list, sam_rho_list, sam_sync_list]))
    # sam_hp_list = sorted(sam_hp_list, key=lambda x: (x[2], x[1],)) # do algs first (unsorted) then rhos and then sync

    seed_list = [x for x in range(5)]

    from train_scripts.cifar5k_minibatch_sam_family_train import train_model
    s = [seed_list, arch_list, sam_hp_list, sgd_hp_list]
    hyp_list = list(itertools.product(*s))

    number_of_tasks = len(hyp_list)
    print(number_of_tasks, "tasks")
    number_of_processes = 4
    # process_ids = [range(number_of_processes)]
    process_ids = [1]
    tasks_to_accomplish = Queue()
    tasks_that_are_done = Queue()
    processes = []

    job = train_model

    for i in range(number_of_tasks):
        tasks_to_accomplish.put(hyp_list[i])

    # creating processes
    for w in process_ids:
        p = Process(target=do_job, args=(tasks_to_accomplish, tasks_that_are_done, job), name=f"Process-N{w+1}")
        processes.append(p)
        p.start()
        time.sleep(100) # slightly longer, was 60 should be 100

    # completing process
    for p in processes:
        p.join()

    # print the output
    while not tasks_that_are_done.empty():
        print(tasks_that_are_done.get())

    return True


if __name__ == '__main__':
    main()
