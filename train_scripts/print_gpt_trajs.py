import glob
import utils
import numpy as np

def main():
    folders = glob.glob("traj/*GPT*")
    folders.sort()
    for folder in folders:
        try:
            mh = utils.load_thing(folder+"/metrics.pkl")
            print(folder)
            print("Train Perp")
            print(np.array(list(mh['train_perplexity'])))
            print("Test Perp")
            print(np.array(list(mh['test_perplexity'])))
            print("Train Acc")
            print(np.array(list(mh['train_accuracy'])))
            print("Test Acc")
            print(np.array(list(mh['test_accuracy'])))
            print("----------------------------------------------------------------------------------------------------")
        except FileNotFoundError:
            pass
        #
    # # mh = utils.load_thing("traj/250502-1158_wiki2_2335_276_GPT2-small-pretrained_seed0_sgdFam_1b0.9_2b0.999_3b0.0_lr0.005_warmup2_wd0.005_bs8/metrics.pkl")
    # mh = utils.load_thing("traj/250502-1403_wiki2_2335_276_stride1024_GPT2-small-pretrained_seed0_sgdFam_1b0.9_2b0.999_3b0.0_lr5e-05_warmup2_wd0.005_bs8/metrics.pkl")
    #


if __name__ == '__main__':
    main()
