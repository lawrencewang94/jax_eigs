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
    process_id = int(current_process().name[-1:])
    os.environ['CUDA_VISIBLE_DEVICES'] = str(process_id - 1)
    os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.9'

    while True:
        try:
            '''
                try to get task from the queue. get_nowait() function will 
                raise queue.Empty exception if the queue is empty. 
                queue(False) function would do the same task also.
            '''
            task_id, task_hyps = tasks_to_accomplish.get_nowait()
            # print(current_process().name)
            # print(int(current_process().name[-1:]))
            out_str, mh, datasets = job(task_hyps, datasets, resume=False, load_only=True)
            print(f"task:{task_id}, process:{process_id}, {out_str}")
            time.sleep(0.5)

            out = str(task_hyps) + f' is done by {current_process().name}; '

            try:
                out += f'test accuracy {mh["test_accuracy"][-1]:%}; lse {mh["lse"]};'
            except KeyError:
                pass

            try:
                ind = np.where(np.array(mh['train_accuracy'][:])>0.999)[0][0]
                out += f" first_99_acc: {mh['test_accuracy'][ind]:%}; cross epoch:[{ind}];"
            except IndexError:
                out += f" first_99_acc: na; cross epoch:[na];"
            except KeyError:
                pass
            # print("out", out)
            tasks_that_are_done.put(out)
            time.sleep(.5)
            del mh

        except queue.Empty:
            del datasets
            del mh
            break

    print(f"{current_process().name} has exited.")
    return True

def main():
    ### Flexible HPs
    arch_list = [
        'vgg',
        'resnet',
    ]

    bs_list = [
        4,
        16,
        64,
        256,
        1024,
        5120
    ]

    sgd_hp_list = [
        # DO NEW -0.1 experiments
        (5e-3, 0., 0.99, -0.1),  # rms-UB
        (5e-3, 0.9, 0.99, -0.1),  # adam-UB

        # FIX ALL EXISTING
        (0.1, 0., 0., 0.),  # sgd
        (5e-3, 0., 0.99, 0.),  # rms
        (5e-3, 0., 0.99, -1.),  # rms-UB
        (5e-3, 0.9, 0.99, 0.),  # adam
        (5e-3, 0.9, 0.99, -1.),  # adam-UB

        # DO -10
        (5e-3, 0., 0.99, -10.),  # rms-UB
        (5e-3, 0.9, 0.99, -10.),  # adam-UB
    ]
    '''
    RETIRED HPS
    (5e-3, 0., 0.9, 0.),  # rms
    (5e-3, 0., 0.9, -1.),  # rms-UB
    (5e-3, 0.9, 0.9, 0.),  # adam
    (5e-3, 0.9, 0.9, -1.),  # adam-UB    
    '''

    # sam_hp_list += list(itertools.product(*[sam_type_list, sam_rho_list, sam_sync_list]))
    # sam_hp_list = sorted(sam_hp_list, key=lambda x: (x[2], x[1],)) # do algs first (unsorted) then rhos and then sync

    seed_list = [x for x in range(5)]

    from train_scripts.cifar5k_ub_family_train import train_model
    s = [sgd_hp_list, seed_list, arch_list, bs_list]
    hyp_list = list(itertools.product(*s))

    number_of_tasks = len(hyp_list)
    print(number_of_tasks, "tasks")
    number_of_processes = 2
    # process_ids = range(number_of_processes)
    process_ids = [2, 3]

    print("--- Starting Processes ---")
    results = []

    # creating processes
    with Pool(processes=number_of_processes) as pool:
        for task_i, task in enumerate(hyp_list):
            result = pool.apply_async(do_job, (task,))  # Non-blocking execution
            results.append(result)

    for result in results:
        print(result.get())  # This blocks until the result is ready


    return True


if __name__ == '__main__':
    main()
