"""@script create_ATR_masks.py
by Benjamin Fildier.

Creates mask for each ATR flight to remove  changes of altitude and changes in direction.
"""


###--- Modules ---###
import os,sys,glob
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import argparse
import pickle

###--- Functions ---###

def getVarMask(var,thres,deriv=False,filt_width=0,center=0,outside=False,above=True):
    """Creates a boolean mask based on a threshold value.
    
    Arguments:
    - var: input array
    - thres: threshold value
    - deriv (default=False): create mask on np.diff(var)
    - filt_width (default=0, no smoothing): width of gaussian filter to be applied before masking
    - center (default=0): reference value when masking outside a range (onlu if outside=True)
    - outside (default=False): mask |X-center| > thres (when above=True) or < thres (when above=False)
    - above (default=True): mask is True above threshold
    
    Returns;
    - boolean array"""

    # smooth if necessary
    values = gaussian_filter(var,filt_width)
    
    # get derivative if necessary
    if deriv:
        values = np.append(np.diff(values),np.nan)
    
    if outside:
        values = np.absolute(values-center)
    
    # take mask
    if above:
        with np.errstate(invalid='ignore'):
            return values > thres
    else:
        with np.errstate(invalid='ignore'):
            return values <= thres

def mergeMasks(mask_list):
    """Merges a list of masks:
    
    Arguments:
    - mask_list: list"""
    
    base_mask = np.zeros(mask_list[0].size,dtype=bool)
    for mask in mask_list:
        
        base_mask = np.logical_or(base_mask,mask)
        
    return base_mask
    
def applyMask(var,mask):
    """Replaces masked values with nans.
    
    Arguments:
    - var: input array
    - mask: boolean array"""
    
    var_out = var.copy()
    var_out[mask] = np.nan
    
    return var_out

def maskAltitudeChange(data,thres=0.1):
    """Create a mask for change in altitude.
    Note: ALTITUDE is smoothed with arbitrary window of 30, and similarly to the 
    default arguments, it was found by trial and error.
    
    Arguments:
    - data: xarray containing variable ALTITUDE
    - thres: threshold value for first derivative of ALTITUDE
    
    Returns: boolean mask"""
    
    # smooth before taking derivative
    z_smooth = gaussian_filter(data.ALTITUDE,30)
    # take derivative
    dz = np.append(np.diff(z_smooth),np.nan)
    # smooth after taking derivative
    dz_smooth = gaussian_filter(dz,30)
    # mask
    mask = getVarMask(dz_smooth,thres,outside=True)
    
    return mask

def maskDirectionChange(data,thres=0.3,thres2=0.008):
    """Create a boolean mask for change of direction.
    Note: HEADING is smoothed with arbitrary window of 60, and similarly to the 
    default arguments, it was found by trial and error.
    
    Arguments:
    - data: xarray containing variable HEADING
    - thres: threshold value for first derivative of HEADING
    - thres2: threshold value for second derivative of HEADING
    
    Returns: boolean mask"""
    
    # fix data for 0deg-360 matching
    var_shift = (data.HEADING+5)%360-5
    var_shift = (var_shift-5)%360+5
    
    # smooth before taking derivative
    dir_smooth = gaussian_filter(var_shift,60)
    # take derivative
    ddir = np.append(np.diff(dir_smooth),np.nan)
    # smooth after taking derivative
    ddir_smooth = gaussian_filter(ddir,0)
    # mask first derivative
    mask_ddir = getVarMask(ddir_smooth,thres,outside=True)
    # second derivative
    dddir = np.append(np.diff(ddir_smooth),np.nan)
    # mask second derivative
    mask_dddir = getVarMask(dddir,thres2,outside=True)
    # merge both masks
    mask = mergeMasks([mask_ddir,mask_dddir])
    
    return mask


##-- Plotting functions --##

def showTbAndZ(data,figfile,between=slice(None),mask=None):
    """Show brightness temperature and altitude.
    
    Arguments:
    - data: data to be shown
    - outputfile: filename to save
    - mask: boolean array
    - between: slice object to crop data in time"""
        
    # mask
    if mask is None:
        mask = np.zeros(data.time.size,dtype=bool)

    # create figure and labels
    fig,ax = plt.subplots(figsize=(15,5))
#     labels = [r'$\lambda = %2.1f \mu m, \nu = %4.0f cm^{-1}$'%(lam,nu) for (lam,nu) in zip(wlengths,wnumbs)]
    wlengths = np.array([8.7,10.8,12.0])
    labels = [r'$\lambda = %2.1f \mu m$'%(lam) for lam in wlengths]
    varids = ['BRIGHTNESS_TEMPERATURE_C1','BRIGHTNESS_TEMPERATURE_C2','BRIGHTNESS_TEMPERATURE_C3']
    
    # show Tb for three canals
    for i in range(3):
        data_show = data[varids[i]].copy()
        data_show[mask] = np.nan
        data_show[between].plot(alpha=0.5,label=labels[i])

    # show altitude
    ax_r = ax.twinx()
    data_show = data.ALTITUDE.copy()
    data_show[mask] = np.nan
    data_show[between].plot(label='height (m)')
    
    # make legend
    ax.legend()
    ax_r.legend(loc='upper right')

    plt.savefig(figfile,bbox_inches='tight')
    plt.close()


###--- Main ---###

if __name__ == '__main__':

    print()
    print('-- WELCOME to create_ATR_masks.py --')

    ##-- Collect arguments --##

    print()
    print('. collect arguments')

    parser = argparse.ArgumentParser(description="Create boolean masks for changes in the course of ATR aircraft.")
    parser.add_argument('-i',"--input_file", type=str,
                        help="Name of ATR file")
    parser.add_argument('--odir',"--outputdir", type=str,
                        help="Directory for output masks")
    parser.add_argument("--make_figs", type=bool, default=False,
                        help="Draw figures?")
    parser.add_argument('--fdir',"--figdir", type=str,
                        help="Directory for output figures")
    args = parser.parse_args()

    ##-- Import data --##

    print()
    print('. open file %s'%args.input_file)
    data = xr.open_dataset(args.input_file)

    ##-- Create masks --##

    print()
    print('. create masks')
    print('   altitude change')
    mask_dz = maskAltitudeChange(data)

    print('   direction change')
    mask_dd = maskDirectionChange(data)

    print('   both')
    mask_all = mergeMasks([mask_dz,mask_dd])

    ##-- Save masks to python objects --##

    print()
    print('. save masks to pickle objects')

    # if not specified define output directory
    if args.odir is None:
        args.odir = os.path.join(os.path.dirname(
            os.path.dirname(os.path.realpath(__file__))),
            'output')
        print('. create output directory %s'%args.odir)
        os.makedirs(args.odir,exist_ok=True)
    
    # MAKE FIGURES

    # save masks
    outputfilename_dz = '%s_mask_alt.pickle'%(os.path.basename(args.input_file).split('.')[0])
    outputfilename_dd = '%s_mask_dir.pickle'%(os.path.basename(args.input_file).split('.')[0])
    outputfilename_all = '%s_mask_alt_dir.pickle'%(os.path.basename(args.input_file).split('.')[0])

    pickle.dump(mask_dz,open(os.path.join(args.odir,outputfilename_dz),'wb'))
    pickle.dump(mask_dd,open(os.path.join(args.odir,outputfilename_dd),'wb'))
    pickle.dump(mask_all,open(os.path.join(args.odir,outputfilename_all),'wb'))

    ##-- Draw figures with different masks --##

    if args.make_figs:

        print()
        print('. make some figures')

         # if not specified define output directory
        if args.fdir is None:
            args.fdir = os.path.join(os.path.dirname(
                os.path.dirname(os.path.realpath(__file__))),
                'figures')
            print('. create figure directory %s'%args.fdir)
            os.makedirs(args.fdir,exist_ok=True)

        # slice to crop in time
        sl = slice(None)
        
        print('show mask for altitude changes')
        figfile_dz = os.path.join(args.fdir,'%s.pdf'%outputfilename_dz.split('.')[0])
        showTbAndZ(data,figfile=figfile_dz,between=sl,mask=mask_dz)
        
        print('show mask for direction changes')
        figfile_dd = os.path.join(args.fdir,'%s.pdf'%outputfilename_dd.split('.')[0])
        showTbAndZ(data,figfile=figfile_dd,between=sl,mask=mask_dd)
        
        print('show mask for both')
        figfile_all = os.path.join(args.fdir,'%s.pdf'%outputfilename_all.split('.')[0])
        showTbAndZ(data,figfile=figfile_all,between=sl,mask=mask_all)

    print()
    print('-- DONE :)')
    sys.exit(0)