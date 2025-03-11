import glob
import os
import shutil

# print("unknown lse")

if __name__ == '__main__':
    root_folder = "traj/"
    # bin_folder = "traj/old/"

    bin_folder = "traj/bin/"
    # folders = glob.glob(root_folder+"*_epochs*") # old version
    folders = glob.glob(root_folder+"*") # new, looking for false negative
    folders.sort(reverse=True) # latest experiments first

    for folder in folders:
        es_files = glob.glob(folder + "/early_stop*")
        if len(es_files) > 0:
            last_save_epoch = int(es_files[0].split("/")[-1][10:-4])
            # break
            if last_save_epoch == 5000:
                print(folder)
                es_file = es_files[0]
                index_epochs_start = folder.find("early_stop")
                # index_epochs_end = folder.find("_bs")
                # new_name = folder[:index_epochs_start] + folder[index_epochs_end:]

                # bin_name = bin_folder + folder.split("/")[-1]
                # shutil.move(folder, bin_name)

            # index_epochs_start = folder.find("_epochs")
            # index_epochs_end = folder.find("_bs")
            # new_name = folder[:index_epochs_start] + folder[index_epochs_end:]
            # bin_name = bin_folder + folder.split("/")[-1]
            # # print("old name", folder, "\n", "new_name", new_name, "\n", "bin name", bin_name)
            # try:
            #     shutil.copytree(folder, new_name)
            # except FileExistsError:
            #     pass
            # shutil.move(folder, bin_name)
            # break