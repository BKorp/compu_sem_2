from pathlib import Path
import pandas as pd
import xml.etree.ElementTree as ET


def token_extractor(xml_file: str, data_dict: dict, lang: str, status: str) -> dict:
    '''Retrieve info from a given xml, return a dict with the info.

    Keyword arguments:
    xml_file -- path string of an xml file
    data_dict -- a dictionary that will hold the data
    lang -- the language of the file
    status -- whether the data is gold, silver or bronze
    '''
    tree = ET.parse(xml_file)
    root = tree.getroot()
    tokens = root.iter('tags')

    for token in tokens:
        data_dict['sent_file'].append(xml_file)
        data_dict['lang'].append(lang)
        data_dict['status'].append(status)
        for tag in token.iter('tag'):
            if tag.attrib['type'] == 'tok':
                data_dict['token'].append(tag.text)
            if tag.attrib['type'] == 'lemma':
                data_dict['lemma'].append(tag.text)
            if tag.attrib['type'] == 'from':
                data_dict['from'].append(tag.text)
            if tag.attrib['type'] == 'to':
                data_dict['to'].append(tag.text)
            if tag.attrib['type'] == 'sem':
                data_dict['semtag'].append(tag.text)


    return data_dict


def file_prepper() -> None:
    '''Expects a pmb file in the same folder as the script,
    sets up the base dict, finds all sentence .xmls, retrieves
    the sentence information for the dict, converts the dict to
    a dataframe, and exports the dataframe as a .csv.
    '''
    data_dict = {
        'sent_file': [],
        'lang': [],
        'status': [],
        'token': [],
        'lemma': [],
        'from': [],
        'to': [],
        'semtag': []
    }
    gold_en = Path('pmb-4.0.0/data/en/gold/')

    for p_num in sorted(gold_en.iterdir()):
        for d_num in sorted(Path(p_num).iterdir()):
            xml_file_gen = Path(d_num).glob('*.xml')
            for xml_file in xml_file_gen:
                data_dict = token_extractor(xml_file, data_dict, 'en', 'gold')

    df = pd.DataFrame.from_dict(data_dict)
    df.to_csv('sem-pmb_4_0_0-gold.csv', index_label='id')


def main():
    file_prepper()


if __name__ == '__main__':
    main()
