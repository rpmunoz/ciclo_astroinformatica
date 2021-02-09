import os
import io
import numpy as np
import matplotlib.pyplot as plt

from astropy import units as u
from astropy.io import fits
from astropy.coordinates import SkyCoord, Angle
from astropy.nddata import Cutout2D
from astropy.wcs import WCS
from astroquery.sdss import SDSS

import requests, urllib
from PIL import Image

absorption_lines={'name':['CaII K','CaII H','Mgb'], 'lambda':[3933.,3968.,5183.], 'align':['right','left','left'], 'offset':[-50,50,50], 'position':[0.05,0.05,0.05], 'color':'r'}
emission_lines={'name':['[O II]',r'H$\beta$',r'H$\alpha$'], 'lambda':[3727.,4861.,6563.], 'align':['left','left','left'], 'offset':[50,50,50], 'position':[0.9,0.9,0.9], 'color':'b'}

def imtoasinh(im_data):
    asinh_percentile=[0.25,99.5]
    asinh_midpoint=-0.07

    im_gray=im_data.copy()
    im_gray = np.nan_to_num(im_gray)
    im_gray_percentile=np.percentile(im_gray,asinh_percentile)
    im_gray_min=im_gray_percentile[0]
    im_gray_max=im_gray_percentile[1]

    print("FITS image - min=%.2f  max=%.2f" % (im_gray_min, im_gray_max))

    im_gray=(im_gray-im_gray_min)/(im_gray_max-im_gray_min)
    im_gray=np.arcsinh(im_gray/asinh_midpoint)/np.arcsinh(1./asinh_midpoint)
    im_gray = np.nan_to_num(im_gray)
    im_gray = np.clip(im_gray, 0., 1.)
    
    return im_gray

def sdss_jpg(coo, fov=1, downloadDir = 'data/sdss'):
    
    try:
        n_coo=len(coo)
    except:
        n_coo=1
        coo=[coo]
        
    if not os.path.exists(downloadDir):
        os.makedirs(downloadDir)
        
    impix = 320
    imsize = fov*u.arcmin
    cutoutbaseurl = 'http://skyservice.pha.jhu.edu/DR12/ImgCutout/getjpeg.aspx'

    n_col=np.min([n_coo,4])
    n_row=np.ceil(n_coo*1./n_col)
    fig, ax = plt.subplots(figsize=(5*n_col,5*n_row), sharex=True, sharey=True)
    for i in range(len(coo)):
        ax = plt.subplot(n_row, n_col, i+1)
        print('Procesando galaxia '+str(i))
        query_string = urllib.parse.urlencode(dict(ra=coo[i].ra.deg, 
                                         dec=coo[i].dec.deg, 
                                         width=impix, height=impix, 
                                         scale=imsize.to(u.arcsec).value/impix))
        url = cutoutbaseurl + '?' + query_string

        im_file=os.path.join(downloadDir, 'SDSS_galaxy{0:02d}'.format(i)+'.jpg')
        urllib.request.urlretrieve(url, im_file)
        im_data = Image.open(im_file)
        
        #r = requests.get(url)
        #im_data=Image.open(io.BytesIO(r.content))
                
        ax.imshow(im_data)
        ax.axis('off')
        ax.text(0.05,0.05, str(i), transform=ax.transAxes, color='white', fontsize=16)
        
    fig.subplots_adjust(hspace=0.1, wspace=0.1)

def sdss_fits(coo, filtro='r', downloadDir = 'data/sdss'):
    
    try:
        n_coo=len(coo)
    except:
        n_coo=1
        coo=[coo]

    n_col=np.min([n_coo,4])
    n_row=np.ceil(n_coo*1./n_col)
    fig, ax = plt.subplots(figsize=(5*n_col,5*n_row), sharex=True, sharey=True)
    for i in range(len(coo)):
        ax = plt.subplot(n_row, n_col, i+1)
        print('Procesando galaxia '+str(i))
        xid = SDSS.query_region(coo[i], spectro=True)
        image=SDSS.get_images(matches=xid, band=filtro)[0][0]

        im_h=image.header
        im_data=image.data
        im_wcs = WCS(im_h)

        im_median=np.median(im_data)
        im_std=np.std(im_data)
        
        im_cutout = Cutout2D(im_data, coo[i], u.Quantity((1., 1.), u.arcmin), wcs=im_wcs)
        im_data = im_cutout.data
        im_wcs = im_cutout.wcs
        im_h = im_wcs.to_header()
        
        im_file = os.path.join(downloadDir, 'SDSS_galaxy{0:02d}_{1:}'.format(i, filtro)+'.fits')

        hdu = fits.PrimaryHDU(im_data, header=im_h)
        hdu.writeto(im_file, overwrite=True)
        
        ax.imshow(im_data,vmin=im_median-1*im_std, vmax=im_median+2*im_std, cmap=plt.get_cmap('gray'), interpolation='nearest', origin='lower')
#        ax.imshow(imtoasinh(im_data.T), vmin=0.1, vmax=0.8, cmap=plt.get_cmap('gray'), interpolation='nearest', origin='lower')
        ax.axis('off')
        ax.text(0.05,0.05, str(i), transform=ax.transAxes, color='white', fontsize=16)
        ax.text(0.95,0.05, 'Filtro '+filtro, transform=ax.transAxes, color='white', fontsize=16, horizontalalignment='right')
        ax.invert_xaxis()
        
    fig.subplots_adjust(hspace=0.1, wspace=0.1)
    
def sdss_spectra(coo, redshift=0., columns=1):
    
    try:
        n_coo=len(coo)
    except:
        n_coo=1
        coo=[coo]

    try:
        n_redshift=len(redshift)
    except:
        n_redshift=1
        redshift=[redshift]


    if n_coo>1 & n_redshift==1:
      redshift=np.ones(n_coo)*redshift[0]

    n_col=np.min([n_coo,columns])
    n_row=np.ceil(n_coo*1./n_col)
    fig, ax = plt.subplots(figsize=(16,6*n_row/(n_col*1.)), sharex=True, sharey=True)
#    fig, ax = plt.subplots(figsize=(20,8*n_coo), sharex=True, sharey=True)
    for i in range(len(coo)):
        ax = plt.subplot(n_row, n_col, i+1)
        #ax = plt.subplot(n_coo, 1,i+1)
        print('Procesando galaxia '+str(i))
        xid = SDSS.query_region(coo[i], spectro=True)
        spec=SDSS.get_spectra(matches=xid)[0][1]

        spec_h=spec.header
        spec_data=spec.data

        loglam=spec_data['loglam']  # Logaritmo de la longitud de onda
        flux=spec_data['flux'] # Flujo medido en unidades de Ergs/cm^2/s/AA

        window_len=9
        s=np.r_[flux[window_len-1:0:-1],flux,flux[-1:-window_len:-1]]
        w=np.ones(window_len,'d')
        w=eval('np.bartlett(window_len)')
        flux_smooth=np.convolve(w/w.sum(),s,mode='valid')[(window_len-1)/2:-(window_len-1)/2]

        gv=(loglam>np.log10(4000.)) & (loglam<np.log10(8000.))
        flux_scale=80./np.percentile(flux_smooth[gv],98)

        #ax.plot(10.**loglam, flux*flux_scale, label=xid['instrument'][0], color='black', linewidth=1)
        ax.plot(10.**loglam, flux_smooth*flux_scale, label=xid['instrument'][0], color='black', linewidth=1)

        for j in range(len(absorption_lines['name'])):
          ax.plot(absorption_lines['lambda'][j]*np.ones(2)*(1.+redshift[i]), [0., 1e5], absorption_lines['color']+'--') 
          ax.text(absorption_lines['lambda'][j]*(1.+redshift[i])+absorption_lines['offset'][j], absorption_lines['position'][j]*100., absorption_lines['name'][j], color=absorption_lines['color'], alpha=0.7, fontsize=16/(n_col*0.8), horizontalalignment=absorption_lines['align'][j])

        for j in range(len(emission_lines['name'])):
          ax.plot(emission_lines['lambda'][j]*np.ones(2)*(1.+redshift[i]), [0., 1e5], emission_lines['color']+'--') 
          ax.text(emission_lines['lambda'][j]*(1.+redshift[i])+emission_lines['offset'][j], emission_lines['position'][j]*100., emission_lines['name'][j], color=emission_lines['color'], alpha=0.7, fontsize=16/(n_col*0.8), horizontalalignment=emission_lines['align'][j])

        if (i % n_col == 0):
            ax.set_ylabel(r'Flujo [10$^{-17}$ ergs/cm$^2$/s/$\AA$]', fontsize=14/(n_col*0.8))
        if (i >= (n_col*(n_row-1))):
            ax.set_xlabel(r'Longitud de onda [$\AA$]', fontsize=14)
        ax.set_title('Galaxia '+str(i))
        ax.set_xlim(3500,8000)
        ax.set_ylim(0.,100.)
        
    fig.subplots_adjust(hspace=0.3, wspace=0.1)


def sdss_template(tipo='eliptica'):

    if tipo=='eliptica':
        template = SDSS.get_spectral_template('galaxy_early')
        title='Espectro de galaxia eliptica'
        lines=absorption_lines
    elif tipo=='espiral': 
        template = SDSS.get_spectral_template('galaxy_late')
        title='Espectro de galaxia espiral'
        lines=emission_lines

    spec_h=template[0][0].header
    spec_data=template[0][0].data
    wcs=WCS(spec_h)  

    index = np.arange(spec_h['NAXIS1'])
    loglam=wcs.wcs_pix2world(index, np.zeros(len(index)), 0)[0]
    flux=spec_data[0]

    gv=(loglam>np.log10(4000.)) & (loglam<np.log10(8000.))
    flux_scale=80./np.max(flux[gv])
  
    fig, ax = plt.subplots(figsize=(20,8), sharex=True, sharey=True)
    plt.plot(10**loglam, flux*flux_scale, '-', color='black', linewidth=1)

    for j in range(len(lines['name'])):
        ax.plot(lines['lambda'][j]*np.ones(2), [0., 1e5], lines['color']+'--')
        ax.text(lines['lambda'][j]+lines['offset'][j], lines['position'][j]*100., lines['name'][j], color='black', fontsize=18, horizontalalignment=lines['align'][j])
  
        ax.set_ylabel(r'Flujo [10$^{-17}$ ergs/cm$^2$/s/\AA]', fontsize=14)
        ax.set_xlabel(r'Longitud de onda [\AA]', fontsize=14)
        ax.set_title(title)
        ax.set_xlim(3500,8000)
        ax.set_ylim(0.,100.)

