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
import logging
import argparse

@contextlib.contextmanager
def nostdout():
    save_stdout = sys.stdout
    sys.stdout = open('trash', 'w')
    yield
    sys.stdout = save_stdout


def setup_logging(log_file):
    """
    Configures logging settings to write logs to the specified file.
    """

    #remove existing file
    if os.path.exists(log_file):
        os.remove(log_file)

    logging.basicConfig(
        filename=log_file,
        filemode='a',  # Append mode
        # format='%(asctime)s - %(levelname)s - %(processName)s - %(message)s',
        format='%(message)s',
        level=logging.WARNING
    )

    print(f"Logging initialized. Writing logs to {log_file}")


def do_job(tasks_to_accomplish, tasks_that_are_done, job):
    supplementary = {}
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
            out_str, mh, supplementary = job(task_hyps, supplementary, resume=False, load_only=False)
            time.sleep(0.5)

            out = f"task:{task_id}, process:{process_id}, {task_hyps}; "

            if 'test_accuracy' in mh and 'lse' in mh:
                out += f'test accuracy {mh["test_accuracy"][-1]:%}; lse {mh["lse"]};'

            if 'train_accuracy' in mh:
                inds = np.where(np.array(mh['train_accuracy'][:])>0.999)[0]
                if len(inds) > 0:
                    ind = inds[0]
                    out += f" first_99_acc: {mh['test_accuracy'][ind]:%}; cross epoch:[{ind}];"

            # print(f"task:{task_id}, process:{process_id}, {task_hyps}")
            # print(out)
            logging.warning(out)
            # tasks_that_are_done.put(out)
            time.sleep(.5)

        except queue.Empty:
            break

    print(f"{current_process().name} has exited.")
    return True

def main():

    parser = argparse.ArgumentParser(description="Run multiprocessing job with optional logging path.")
    parser.add_argument("--log", type=str, default="txt/training_log.txt",
                        help="Specify the log file path (default: training_log.txt)")
    args = parser.parse_args()

    setup_logging(args.log)

    ### Flexible HPs

    arch_list = [
        'resnet',
    ]

    bs_list = [
        16,
    ]

    sgd_hp_list = []
    for i in range(1, 10):
        sgd_hp_list.append((4e-3, 0., 0.99, -utils.signif(i * 0.1, 3)))
    for i in range(1, 11):
        sgd_hp_list.append((4e-3, 0., 0.99, -utils.signif(i * 1., 3)))


    # sgd_hp_list = [
    #     (4e-3, 0., 0.99, 0.0),  # baseline rms # tunes to 0.004 or 0.005
    #     # (4e-3, 0., 0.99, 0.0),  # baseline rms # tunes to 0.004 or 0.005
    # ]

    # sgd_hp_list = [

        # (6e-3, 0., 0.99, 0.0),  # baseline rms

        #
        # (6e-3, 0., 0.99, -1e+3),  # rms-UB
        # (6e-3, 0., 0.99, -1e+2),  # rms-UB
        # (6e-3, 0., 0.99, -1e+1),  # rms-UB
        # (6e-3, 0., 0.99, -1e-0),  # rms-UB
        # (6e-3, 0., 0.99, -1e-1),  # rms-UB
        # (6e-3, 0., 0.99, -1e-2),  # rms-UB
        # (6e-3, 0., 0.99, -1e-3),  # rms-UB

    # ]


    seed_list = [x for x in range(3)]

    from train_scripts.cifar5k_rms_tune_train import train_model
    s = [seed_list, arch_list, bs_list, sgd_hp_list]
    hyp_list = list(itertools.product(*s))

    number_of_tasks = len(hyp_list)
    print(f"{number_of_tasks} tasks to process")
    logging.warning(f"{number_of_tasks} tasks to process")

    number_of_processes = 4
    process_ids = range(number_of_processes)
    # process_ids = [2, 3]
    tasks_to_accomplish = Queue()
    tasks_that_are_done = Queue()
    processes = []

    job = train_model

    for i in range(number_of_tasks):
        tasks_to_accomplish.put((i, hyp_list[i]))

    print("--- Starting Processes ---")
    # creating processes
    for w in process_ids:
        p = Process(target=do_job, args=(tasks_to_accomplish, tasks_that_are_done, job), name=f"Process-N{w+1}")
        processes.append(p)

    for p in processes:
        print(f"Starting: {p.name}")
        p.start()
        time.sleep(0.5)

    print("--- Joining Processes ---")
    # completing process
    for p in processes:
        p.join()
        print(f"{p.name} joined. ")

    # print("--- Flushing Outputs ---")
    # # print the output
    # while not tasks_that_are_done.empty():
    #     print(tasks_that_are_done.get())

    return True


if __name__ == '__main__':
    main()
