"""@script create_ATR_masks_allflights.py
by B. Fildier

Create mask for change in altitude and direction of ATR for all ATR flights.
"""

import glob,os,sys

if __name__ == '__main__':

    currentdir = os.path.dirname(os.path.realpath(__file__))
    inputdir = os.path.join(os.path.dirname(currentdir),'input')

    files = glob.glob(os.path.join(inputdir,'*.nc'))

    for file in files:

        print('-----------------------------------------------------------------')
        print('######### %s ##########'%os.path.basename(file))
        print('-----------------------------------------------------------------')

        os.system('python create_ATR_masks.py -i %s --make_figs True'%file)
        
        print()

    sys.exit(0)