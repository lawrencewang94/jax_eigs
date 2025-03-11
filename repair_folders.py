import glob
import os
import shutil

# print("unknown lse")

if __name__ == '__main__':
    root_folder = "traj/"

    bin_folder = "traj/bin/"
    folders = glob.glob(root_folder+"*") # new, looking for false negative
    folders.sort(reverse=True) # latest experiments first

    for folder in folders:
        es_files = glob.glob(folder + "/early_stop*")
        if len(es_files) > 0:
            last_save_epoch = int(es_files[0].split("/")[-1][10:-4])
            # break
            if last_save_epoch == 5000:
                # print(folder)
                es_file = es_files[0]
                index_es = es_file.find("early_stop")
                # index_epochs_end = folder.find("_bs")
                new_name = es_file[:index_es] + "no_converge" + es_file[index_es+10:]
                # print(es_file, "\n", new_name)
                # break
                shutil.move(es_file, new_name)
