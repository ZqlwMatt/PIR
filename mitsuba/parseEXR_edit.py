"""
OpenEXR file parsing tool
Dependencies: openexr, openexr-python
"""
import OpenEXR
import argparse
import numpy as np
import os
from imageio import imsave
import sys

def saveResultsCompact(name, results, width):
    intv = 5
    num = len(results) 
    w_n = int(width)
    h_n = int(np.ceil(float(num) / w_n))
    big_img = np.zeros(0)
    fix_h = 0; fix_w = 0
    for count, img in enumerate(results):
        if isinstance(img, bool) and img == False:
            print('Skipping False result')
            continue
        if img.ndim < 3:
            img = np.tile(img.reshape(img.shape[0], img.shape[1], 1), (1,1,3))
        h, w, c = img.shape
        if count == 0:
           big_img = np.zeros((h_n*h + (h_n-1)*intv, w_n*w + (w_n-1)*intv, 3), dtype=np.uint8)
           fix_h = h
           fix_w = w
        h_idx = int(count / w_n)
        w_idx = count % w_n
        h_start = h_idx * (h + intv)
        w_start = w_idx * (w + intv)
        big_img[h_start:h_start+h, w_start:w_start+w, :] = img
    imsave(name, big_img)

def toArray(I, cname, unmute=True):
    hw = I.header()['dataWindow']
    w = hw.max.x+1
    h = hw.max.y+1
    print(cname)
    
    if cname != 'depth':
        prefix = cname + '.' if cname + '.R' in I.header()['channels'].keys() else ''
        r = I.channel(prefix + 'R')
        g = I.channel(prefix + 'G')
        b = I.channel(prefix + 'B')
        R = np.frombuffer(r, np.float16).reshape(h, w)
        G = np.frombuffer(g, np.float16).reshape(h, w)
        B = np.frombuffer(b, np.float16).reshape(h, w)

        img = np.stack([R, G, B], 0).astype(np.float16)
        img = img.transpose(1, 2, 0)
    else:
        prefix = cname + '.' if cname + '.Y' in I.header()['channels'].keys() else ''
        y = I.channel(prefix + 'Y')
        img = np.frombuffer(y, np.float16).reshape(h, w)
    
    if unmute:
        print('[Before normalization %s], max value: %f, min: %f, mean: %f' % 
                (cname, img.max(), img.min(), img.mean()))
    
    if cname == 'normal':
        img = (img + 1) * 0.5 * 255
    elif cname == 'depth':
        img = (img / img.max()) * 255
    else:  # albedo, color
        img = (img * 255).clip(0, 255)
    
    return img

def getSavePrefix(out_dir, in_file, suffix):
    prefix, ext = os.path.splitext(os.path.basename(in_file))
    save_prefix = os.path.join(out_dir, prefix + suffix)
    return save_prefix

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract and process data from OpenEXR files')
    parser.add_argument('in_file', help='Input EXR file')
    parser.add_argument('--out_dir', default='./', help='Output directory')
    parser.add_argument('--unmute', action='store_true', default=False, help='Enable verbose output')
    parser.add_argument('--get_img', action='store_true', default=False, help='Extract image data')
    parser.add_argument('--do_img_gamma', action='store_true', default=False, help='Apply gamma correction to image')
    parser.add_argument('--get_normal', action='store_true', default=False, help='Extract normal data')
    parser.add_argument('--get_albedo', action='store_true', default=False, help='Extract albedo data')
    parser.add_argument('--get_depth', action='store_true', default=False, help='Extract depth data')
    parser.add_argument('--compact', action='store_true', default=False, 
                help='Save all components in a single figure for visualization')
    parser.add_argument('--suffix', default='', help='Suffix for output filenames')
    args = parser.parse_args()
    
    if args.unmute:
        print(args)

    I = OpenEXR.InputFile(args.in_file)

    channels = I.header()['channels'].keys()
    if args.unmute:
        print(channels)

    save_prefix = getSavePrefix(args.out_dir, args.in_file, args.suffix)
    compact = []
    
    if args.get_normal:
        normal = toArray(I, 'normal', args.unmute)
        if args.compact: 
            compact.append(normal.astype(np.uint8))
        else:
            imsave(save_prefix + '_normal.png', normal.astype(np.uint8))
    
    if args.get_depth:
        depth = toArray(I, 'depth', args.unmute)
        if args.compact: 
            compact.append(depth.astype(np.uint8))
        else:
            imsave(save_prefix + '_depth.png', depth.astype(np.uint8))
    
    if args.get_img:
        img = toArray(I, 'img', args.unmute)
        if args.do_img_gamma:
            img = np.power(img/255.0 + 1e-6, 1./2.2) * 255.0
        if args.compact: 
            compact.append(img.astype(np.uint8))
        else:
            imsave(save_prefix + '_img.png', img.astype(np.uint8))
    
    if args.get_albedo:
        img = toArray(I, 'albedo', args.unmute)
        if args.compact: 
            compact.append(img.astype(np.uint8))
        else:
            imsave(save_prefix + '_albedo.png', img.astype(np.uint8))
    
    if args.compact and compact:
        saveResultsCompact(save_prefix + '.jpg', compact, len(compact))
    
    if args.unmute:
        print('[Prefix] %s, infile: %s' % (save_prefix, args.in_file))
        print(I.header().keys())
        if args.get_normal:
            print('[Normal] Max %f, Min %f, Mean %f' % (normal.max(), normal.min(), normal.mean()))
        if args.get_img:
            print('[Img] Max %f, Min %f, Mean %f' % (img.max(), img.min(), img.mean()))
