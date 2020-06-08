# -*- coding: utf-8 -*-
"""
Created on Sun Jun  7 15:41:48 2020

@author: Anirudh
"""


import os
import glob
import pandas as pd
import xml.etree.ElementTree as ET


def xml_to_csv(path):
    xml_list = []
    for xml_file in glob.glob(path + '/*.xml'):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall('object'):
            value = [root.find('filename').text,
                     int(member[4][0].text),
                     int(member[4][1].text),
                     int(member[4][2].text),
                     int(member[4][3].text)
                     ]
            if(member[0].text=='patchy roads'):
                value.extend('5')
            
            elif(member[0].text=='overflowing garbage'):
                value.extend('4')
            
            elif(member[0].text=='proper manhole'):
                value.extend('6')

            elif(member[0].text=='Proper garbage'):
                value.extend('3')
                
            elif(member[0].text=='Proper road'):
                value.extend('2')
                
            else:
                value.extend('1')
                
            xml_list.append(value)
            
                
    column_name = ['filename', 'xmin', 'ymin', 'xmax', 'ymax','class']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    return xml_df


def main():
    image_path = os.path.join(os.getcwd(), 'val')
    xml_df = xml_to_csv(image_path)
    xml_df.to_csv('val_data.csv', index=None)
    print('Successfully converted xml to csv.')


main()