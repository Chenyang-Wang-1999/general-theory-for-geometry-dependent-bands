'''
author:        Wang Chenyang <cy-wang21@mails.tsinghua.edu.cn>
date:          2025-03-10
Copyright © Department of Physics, Tsinghua University. All rights reserved
'''

import re
import os
import shutil


def get_graphics(curr_str):
    ''' Extract the graphics from the current string '''
    graphics = re.findall(r'\\includegraphics(?:\[[^\]]*\])?{([^}]+)}', curr_str)
    return graphics


def get_bibfile(curr_str):
    ''' Extract the bibfile from the current string '''
    bibfile = re.findall(r'\\bibliography{([^}]+)}', curr_str)
    return bibfile


def get_namelist(fname: str):
    ''' Extract the namelist from the given file '''
    graphics_list = []
    bib_list = []
    with open(fname, 'r', encoding='utf-8') as f:
        for line in f:
            any_graphics = get_graphics(line)
            any_bibfile = get_bibfile(line)
            if any_graphics or any_bibfile:
                graphics_list.extend(any_graphics)
                bib_list.extend(any_bibfile)
    return graphics_list, bib_list


def pack_tex(tex_fname, target_folder: str):
    ''' Pack the tex file to the target folder'''
    # open tex file by file explorer
    if os.path.exists(target_folder):
        shutil.rmtree(target_folder)
    os.makedirs(target_folder)
    graphics_list, bib_list = get_namelist(tex_fname)
    for curr_prefix in graphics_list:
        folder_list = curr_prefix.split('/')[:-1]
        for curr_suffix in ['svg', 'pdf']:
            curr_file = curr_prefix + '.' + curr_suffix
            curr_folder = ""
            for folder in folder_list:
                curr_folder = os.path.join(curr_folder, folder)
                if not os.path.exists(os.path.join(target_folder, curr_folder)):
                    os.mkdir(os.path.join(target_folder, curr_folder))
            if os.path.exists(curr_file):
                print(curr_folder)
                print(os.path.join(target_folder, curr_folder))
                print(curr_file)
                shutil.copy(curr_file, os.path.join(target_folder, curr_folder))
    for curr_bib in bib_list:
        curr_bib = curr_bib + '.bib'
        if os.path.exists(curr_bib):
            shutil.copy(curr_bib, target_folder)
    shutil.copy(tex_fname, target_folder)


if __name__ == '__main__':
    # namelist = get_namelist('manuscript-and-supplementary-materials-v2.tex')
    pack_tex("manuscript-and-supplementary-materials-main.tex", "./ver-PhysRevX")