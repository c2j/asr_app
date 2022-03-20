# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#import argparse
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import io
import os
import sys
from typing import List
from typing import Optional
from typing import Union

import librosa
import numpy as np
import paddle
import soundfile
import yaml
from yacs.config import CfgNode

#from ..executor import BaseExecutor
#from ..log import logger
import logging 
#from ..utils import cli_register
#from ..utils import download_and_decompress
#from ..utils import MODEL_HOME

from paddlespeech.s2t.frontend.featurizer.text_featurizer import TextFeaturizer
from paddlespeech.s2t.transform.transformation import Transformation
from paddlespeech.s2t.utils.dynamic_import import dynamic_import
from paddlespeech.s2t.utils.utility import UpdateConfig


import os
import tarfile
import zipfile
from typing import Any
from typing import Dict

from paddle.framework import load

#from . import download
#from .entry import commands





import hashlib
import os
import os.path as osp
import shutil
import subprocess
import tarfile
import time
import zipfile

import requests
from tqdm import tqdm

import re
from typing import List
from typing import Optional
from typing import Union

from datetime import datetime, timedelta


DOWNLOAD_RETRY_LIMIT = 3

logging.basicConfig(filename='logger.log', level=logging.ERROR)
logger = logging.getLogger("ASR")

def _is_url(path):
    """
    Whether path is URL.
    Args:
        path (string): URL string or not.
    """
    return path.startswith('http://') or path.startswith('https://')


def _map_path(url, root_dir):
    # parse path after download under root_dir
    fname = osp.split(url)[-1]
    fpath = fname
    return osp.join(root_dir, fpath)


def _get_unique_endpoints(trainer_endpoints):
    # Sorting is to avoid different environmental variables for each card
    trainer_endpoints.sort()
    ips = set()
    unique_endpoints = set()
    for endpoint in trainer_endpoints:
        ip = endpoint.split(":")[0]
        if ip in ips:
            continue
        ips.add(ip)
        unique_endpoints.add(endpoint)
    logger.info("unique_endpoints {}".format(unique_endpoints))
    return unique_endpoints


def get_path_from_url(url,
                      root_dir,
                      md5sum=None,
                      check_exist=True,
                      decompress=True,
                      method='get'):
    """ Download from given url to root_dir.
    if file or directory specified by url is exists under
    root_dir, return the path directly, otherwise download
    from url and decompress it, return the path.
    Args:
        url (str): download url
        root_dir (str): root dir for downloading, it should be
                        WEIGHTS_HOME or DATASET_HOME
        md5sum (str): md5 sum of download package
        decompress (bool): decompress zip or tar file. Default is `True`
        method (str): which download method to use. Support `wget` and `get`. Default is `get`.
    Returns:
        str: a local path to save downloaded models & weights & datasets.
    """

    from paddle.fluid.dygraph.parallel import ParallelEnv

    assert _is_url(url), "downloading from {} not a url".format(url)
    # parse path after download to decompress under root_dir
    fullpath = _map_path(url, root_dir)
    # Mainly used to solve the problem of downloading data from different 
    # machines in the case of multiple machines. Different ips will download 
    # data, and the same ip will only download data once.
    unique_endpoints = _get_unique_endpoints(ParallelEnv().trainer_endpoints[:])
    if osp.exists(fullpath) and check_exist and _md5check(fullpath, md5sum):
        logger.info("Found {}".format(fullpath))
    else:
        if ParallelEnv().current_endpoint in unique_endpoints:
            fullpath = _download(url, root_dir, md5sum, method=method)
        else:
            while not os.path.exists(fullpath):
                time.sleep(1)

    if ParallelEnv().current_endpoint in unique_endpoints:
        if decompress and (tarfile.is_tarfile(fullpath) or
                           zipfile.is_zipfile(fullpath)):
            fullpath = _decompress(fullpath)

    return fullpath


def _get_download(url, fullname):
    # using requests.get method
    fname = osp.basename(fullname)
    try:
        req = requests.get(url, stream=True)
    except Exception as e:  # requests.exceptions.ConnectionError
        logger.info("Downloading {} from {} failed with exception {}".format(
            fname, url, str(e)))
        return False

    if req.status_code != 200:
        raise RuntimeError("Downloading from {} failed with code "
                           "{}!".format(url, req.status_code))

    # For protecting download interupted, download to
    # tmp_fullname firstly, move tmp_fullname to fullname
    # after download finished
    tmp_fullname = fullname + "_tmp"
    total_size = req.headers.get('content-length')
    with open(tmp_fullname, 'wb') as f:
        if total_size:
            with tqdm(total=(int(total_size) + 1023) // 1024) as pbar:
                for chunk in req.iter_content(chunk_size=1024):
                    f.write(chunk)
                    pbar.update(1)
        else:
            for chunk in req.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
    shutil.move(tmp_fullname, fullname)

    return fullname


def _wget_download(url, fullname):
    # using wget to download url
    tmp_fullname = fullname + "_tmp"
    # –user-agent
    command = 'wget -O {} -t {} {}'.format(tmp_fullname, DOWNLOAD_RETRY_LIMIT,
                                           url)
    subprc = subprocess.Popen(
        command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    _ = subprc.communicate()

    if subprc.returncode != 0:
        raise RuntimeError(
            '{} failed. Please make sure `wget` is installed or {} exists'.
            format(command, url))

    shutil.move(tmp_fullname, fullname)

    return fullname


_download_methods = {
    'get': _get_download,
    'wget': _wget_download,
}


def _download(url, path, md5sum=None, method='get'):
    """
    Download from url, save to path.
    url (str): download url
    path (str): download to given path
    md5sum (str): md5 sum of download package
    method (str): which download method to use. Support `wget` and `get`. Default is `get`.
    """
    assert method in _download_methods, 'make sure `{}` implemented'.format(
        method)

    if not osp.exists(path):
        os.makedirs(path)

    fname = osp.split(url)[-1]
    fullname = osp.join(path, fname)
    retry_cnt = 0

    logger.info("Downloading {} from {}".format(fname, url))
    while not (osp.exists(fullname) and _md5check(fullname, md5sum)):
        if retry_cnt < DOWNLOAD_RETRY_LIMIT:
            retry_cnt += 1
        else:
            raise RuntimeError("Download from {} failed. "
                               "Retry limit reached".format(url))

        if not _download_methods[method](url, fullname):
            time.sleep(1)
            continue

    return fullname


def _md5check(fullname, md5sum=None):
    if md5sum is None:
        return True

    logger.info("File {} md5 checking...".format(fullname))
    md5 = hashlib.md5()
    with open(fullname, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            md5.update(chunk)
    calc_md5sum = md5.hexdigest()

    if calc_md5sum != md5sum:
        logger.info("File {} md5 check failed, {}(calc) != "
                    "{}(base)".format(fullname, calc_md5sum, md5sum))
        return False
    return True


def _decompress(fname):
    """
    Decompress for zip and tar file
    """
    logger.info("Decompressing {}...".format(fname))

    # For protecting decompressing interupted,
    # decompress to fpath_tmp directory firstly, if decompress
    # successed, move decompress files to fpath and delete
    # fpath_tmp and remove download compress file.

    if tarfile.is_tarfile(fname):
        uncompressed_path = _uncompress_file_tar(fname)
    elif zipfile.is_zipfile(fname):
        uncompressed_path = _uncompress_file_zip(fname)
    else:
        raise TypeError("Unsupport compress file type {}".format(fname))

    return uncompressed_path


def _uncompress_file_zip(filepath):
    files = zipfile.ZipFile(filepath, 'r')
    file_list = files.namelist()

    file_dir = os.path.dirname(filepath)

    if _is_a_single_file(file_list):
        rootpath = file_list[0]
        uncompressed_path = os.path.join(file_dir, rootpath)

        for item in file_list:
            files.extract(item, file_dir)

    elif _is_a_single_dir(file_list):
        rootpath = os.path.splitext(file_list[0])[0].split(os.sep)[0]
        uncompressed_path = os.path.join(file_dir, rootpath)

        for item in file_list:
            files.extract(item, file_dir)

    else:
        rootpath = os.path.splitext(filepath)[0].split(os.sep)[-1]
        uncompressed_path = os.path.join(file_dir, rootpath)
        if not os.path.exists(uncompressed_path):
            os.makedirs(uncompressed_path)
        for item in file_list:
            files.extract(item, os.path.join(file_dir, rootpath))

    files.close()

    return uncompressed_path


def _uncompress_file_tar(filepath, mode="r:*"):
    files = tarfile.open(filepath, mode)
    file_list = files.getnames()

    file_dir = os.path.dirname(filepath)

    if _is_a_single_file(file_list):
        rootpath = file_list[0]
        uncompressed_path = os.path.join(file_dir, rootpath)
        for item in file_list:
            files.extract(item, file_dir)
    elif _is_a_single_dir(file_list):
        rootpath = os.path.splitext(file_list[0])[0].split(os.sep)[-1]
        uncompressed_path = os.path.join(file_dir, rootpath)
        for item in file_list:
            files.extract(item, file_dir)
    else:
        rootpath = os.path.splitext(filepath)[0].split(os.sep)[-1]
        uncompressed_path = os.path.join(file_dir, rootpath)
        if not os.path.exists(uncompressed_path):
            os.makedirs(uncompressed_path)

        for item in file_list:
            files.extract(item, os.path.join(file_dir, rootpath))

    files.close()

    return uncompressed_path


def _is_a_single_file(file_list):
    if len(file_list) == 1 and file_list[0].find(os.sep) < -1:
        return True
    return False


def _is_a_single_dir(file_list):
    new_file_list = []
    for file_path in file_list:
        if '/' in file_path:
            file_path = file_path.replace('/', os.sep)
        elif '\\' in file_path:
            file_path = file_path.replace('\\', os.sep)
        new_file_list.append(file_path)

    file_name = new_file_list[0].split(os.sep)[0]
    for i in range(1, len(new_file_list)):
        if file_name != new_file_list[i].split(os.sep)[0]:
            return False
    return True



def cli_register(name: str, description: str='') -> Any:
    def _warpper(command):
        items = name.split('.')

        com = commands
        for item in items:
            com = com[item]
        com['_entry'] = command
        if description:
            com['_description'] = description
        return command

    return _warpper


def get_command(name: str) -> Any:
    items = name.split('.')
    com = commands
    for item in items:
        com = com[item]

    return com['_entry']


def _get_uncompress_path(filepath: os.PathLike) -> os.PathLike:
    file_dir = os.path.dirname(filepath)
    is_zip_file = False
    if tarfile.is_tarfile(filepath):
        files = tarfile.open(filepath, "r:*")
        file_list = files.getnames()
    elif zipfile.is_zipfile(filepath):
        files = zipfile.ZipFile(filepath, 'r')
        file_list = files.namelist()
        is_zip_file = True
    else:
        return file_dir

    if _is_a_single_file(file_list):
        rootpath = file_list[0]
        uncompressed_path = os.path.join(file_dir, rootpath)
    elif _is_a_single_dir(file_list):
        if is_zip_file:
            rootpath = os.path.splitext(file_list[0])[0].split(os.sep)[0]
        else:
            rootpath = os.path.splitext(file_list[0])[0].split(os.sep)[-1]
        uncompressed_path = os.path.join(file_dir, rootpath)
    else:
        rootpath = os.path.splitext(filepath)[0].split(os.sep)[-1]
        uncompressed_path = os.path.join(file_dir, rootpath)

    files.close()
    return uncompressed_path


def download_and_decompress(archive: Dict[str, str], path: str) -> os.PathLike:
    """
    Download archieves and decompress to specific path.
    """
    print("For Debug: "+path)
    if not os.path.isdir(path):
        os.makedirs(path)

    assert 'url' in archive and 'md5' in archive, \
        'Dictionary keys of "url" and "md5" are required in the archive, but got: {}'.format(list(archive.keys()))

    filepath = os.path.join(path, os.path.basename(archive['url']))
    if os.path.isfile(filepath) and _md5check(filepath,
                                                       archive['md5']):
        uncompress_path = _get_uncompress_path(filepath)
        if not os.path.isdir(uncompress_path):
            _decompress(filepath)
    else:
        uncompress_path = get_path_from_url(archive['url'], path,
                                                     archive['md5'])

    return uncompress_path


def load_state_dict_from_url(url: str, path: str, md5: str=None) -> os.PathLike:
    """
    Download and load a state dict from url
    """
    if not os.path.isdir(path):
        os.makedirs(path)

    get_path_from_url(url, path, md5)
    return load(os.path.join(path, os.path.basename(url)))


def _get_user_home():
    return os.path.expanduser('~')


def _get_paddlespcceh_home():
    if 'PPSPEECH_HOME' in os.environ:
        home_path = os.environ['PPSPEECH_HOME']
        if os.path.exists(home_path):
            if os.path.isdir(home_path):
                return home_path
            else:
                raise RuntimeError(
                    'The environment variable PPSPEECH_HOME {} is not a directory.'.
                    format(home_path))
        else:
            return home_path
    return os.path.join(_get_user_home(), '.paddlespeech')


def _get_sub_home(directory):
    home = os.path.join(_get_paddlespcceh_home(), directory)
    if not os.path.exists(home):
        os.makedirs(home)
    return home


PPSPEECH_HOME = _get_paddlespcceh_home()
MODEL_HOME = _get_sub_home('models')
#MODEL_HOME = "/Users/c2j/.paddlespeech"


def singleton(cls):
    _instance = {}

    def inner():
        if cls not in _instance:
            _instance[cls] = cls()
        return _instance[cls]
    return inner

pretrained_models = {
    # The tags for pretrained_models should be "{model_name}[_{dataset}][-{lang}][-...]".
    # e.g. "conformer_wenetspeech-zh-16k" and "panns_cnn6-32k".
    # Command line and python api use "{model_name}[_{dataset}]" as --model, usage:
    # "paddlespeech asr --model conformer_wenetspeech --lang zh --sr 16000 --input ./input.wav"
    "conformer_wenetspeech-zh-16k": {
        'url':
        'https://paddlespeech.bj.bcebos.com/s2t/wenetspeech/conformer.model.tar.gz',
        'md5':
        '54e7a558a6e020c2f5fb224874943f97',
        'cfg_path':
        'conf/conformer.yaml',
        'ckpt_path':
        'exp/conformer/checkpoints/wenetspeech',
    },
    #Text
    "ernie_linear_p7_wudao-punc-zh": {
        'url':
        'https://paddlespeech.bj.bcebos.com/text/ernie_linear_p7_wudao-punc-zh.tar.gz',
        'md5':
        '12283e2ddde1797c5d1e57036b512746',
        'cfg_path':
        'ckpt/model_config.json',
        'ckpt_path':
        'ckpt/model_state.pdparams',
        'vocab_file':
        'punc_vocab.txt',
    },
    "ernie_linear_p3_wudao-punc-zh": {
        'url':
        'https://paddlespeech.bj.bcebos.com/text/ernie_linear_p3_wudao-punc-zh.tar.gz',
        'md5':
        '448eb2fdf85b6a997e7e652e80c51dd2',
        'cfg_path':
        'ckpt/model_config.json',
        'ckpt_path':
        'ckpt/model_state.pdparams',
        'vocab_file':
        'punc_vocab.txt',
    },
}

model_alias = {
    "ds2_offline": "paddlespeech.s2t.models.ds2:DeepSpeech2Model",
    "ds2_online": "paddlespeech.s2t.models.ds2_online:DeepSpeech2ModelOnline",
    "conformer": "paddlespeech.s2t.models.u2:U2Model",
    "transformer": "paddlespeech.s2t.models.u2:U2Model",
    "wenetspeech": "paddlespeech.s2t.models.u2:U2Model",
    #Text
    "ernie_linear_p7": "paddlespeech.text.models:ErnieLinear",
    "ernie_linear_p3": "paddlespeech.text.models:ErnieLinear",
}

tokenizer_alias = {
    "ernie_linear_p7": "paddlenlp.transformers:ErnieTokenizer",
    "ernie_linear_p3": "paddlenlp.transformers:ErnieTokenizer",
}

#@cli_register(
#    name='paddlespeech.asr', description='Speech to text infer command.')
@singleton
class ASRExecutor():
    def __init__(self):
        #super(ASRExecutor, self).__init__()
        self._inputs = dict()
        self._outputs = dict()
        

    def _get_pretrained_path(self, tag: str) -> os.PathLike:
        """
        Download and returns pretrained resources path of current task.
        """
        assert tag in pretrained_models, 'Can not find pretrained resources of {}.'.format(
            tag)

        res_path = os.path.join(MODEL_HOME, tag)
        decompressed_path = download_and_decompress(pretrained_models[tag],
                                                    res_path)
        decompressed_path = os.path.abspath(decompressed_path)
        logger.info(
            'Use pretrained model stored in: {}'.format(decompressed_path))

        return decompressed_path

    def _init_from_path(self,
                        model_type: str='wenetspeech',
                        lang: str='zh',
                        sample_rate: int=16000,
                        cfg_path: Optional[os.PathLike]=None,
                        ckpt_path: Optional[os.PathLike]=None):
        """。
        Init model and other resources from a specific path.
        """
        if hasattr(self, 'model'):
            logger.info('Model had been initialized.')
            return

        if cfg_path is None or ckpt_path is None:
            sample_rate_str = '16k' if sample_rate == 16000 else '8k'
            tag = model_type + '-' + lang + '-' + sample_rate_str
            res_path = self._get_pretrained_path(tag)  # wenetspeech_zh
            self.res_path = res_path
            self.cfg_path = os.path.join(res_path,
                                         pretrained_models[tag]['cfg_path'])
            self.ckpt_path = os.path.join(
                res_path, pretrained_models[tag]['ckpt_path'] + ".pdparams")
            logger.info(res_path)
            logger.info(self.cfg_path)
            logger.info(self.ckpt_path)
        else:
            self.cfg_path = os.path.abspath(cfg_path)
            self.ckpt_path = os.path.abspath(ckpt_path + ".pdparams")
            res_path = os.path.dirname(
                os.path.dirname(os.path.abspath(self.cfg_path)))

        #Init body.
        self.config = CfgNode(new_allowed=True)
        self.config.merge_from_file(self.cfg_path)
        self.config.decoding.decoding_method = "attention_rescoring"

        with UpdateConfig(self.config):
            if "ds2_online" in model_type or "ds2_offline" in model_type:
                from paddlespeech.s2t.io.collator import SpeechCollator
                self.config.collator.vocab_filepath = os.path.join(
                    res_path, self.config.collator.vocab_filepath)
                self.config.collator.mean_std_filepath = os.path.join(
                    res_path, self.config.collator.cmvn_path)
                self.collate_fn_test = SpeechCollator.from_config(self.config)
                text_feature = TextFeaturizer(
                    unit_type=self.config.collator.unit_type,
                    vocab=self.config.collator.vocab_filepath,
                    spm_model_prefix=self.config.collator.spm_model_prefix)
                self.config.model.input_dim = self.collate_fn_test.feature_size
                self.config.model.output_dim = text_feature.vocab_size
            elif "conformer" in model_type or "transformer" in model_type or "wenetspeech" in model_type:
                self.config.collator.vocab_filepath = os.path.join(
                    res_path, self.config.collator.vocab_filepath)
                self.config.collator.augmentation_config = os.path.join(
                    res_path, self.config.collator.augmentation_config)
                self.config.collator.spm_model_prefix = os.path.join(
                    res_path, self.config.collator.spm_model_prefix)
                text_feature = TextFeaturizer(
                    unit_type=self.config.collator.unit_type,
                    vocab=self.config.collator.vocab_filepath,
                    spm_model_prefix=self.config.collator.spm_model_prefix)
                self.config.model.input_dim = self.config.collator.feat_dim
                self.config.model.output_dim = text_feature.vocab_size

            else:
                raise Exception("wrong type")
        # Enter the path of model root

        model_name = model_type[:model_type.rindex(
            '_')]  # model_type: {model_name}_{dataset}
        model_class = dynamic_import(model_name, model_alias)
        model_conf = self.config.model
        logger.info(model_conf)
        model = model_class.from_config(model_conf)
        self.model = model
        self.model.eval()

        # load model
        model_dict = paddle.load(self.ckpt_path)
        self.model.set_state_dict(model_dict)

    def preprocess(self, model_type: str, input: Union[str, os.PathLike]):
        """
        Input preprocess and return paddle.Tensor stored in self.input.
        Input content can be a text(tts), a file(asr, cls) or a streaming(not supported yet).
        """

        audio_file = input
        logger.info("Preprocess audio_file:" + audio_file)

        # Get the object for feature extraction
        if "ds2_online" in model_type or "ds2_offline" in model_type:
            audio, _ = self.collate_fn_test.process_utterance(
                audio_file=audio_file, transcript=" ")
            audio_len = audio.shape[0]
            audio = paddle.to_tensor(audio, dtype='float32')
            audio_len = paddle.to_tensor(audio_len)
            audio = paddle.unsqueeze(audio, axis=0)
            vocab_list = collate_fn_test.vocab_list
            self._inputs["audio"] = audio
            self._inputs["audio_len"] = audio_len
            logger.info(f"audio feat shape: {audio.shape}")

        elif "conformer" in model_type or "transformer" in model_type or "wenetspeech" in model_type:
            logger.info("get the preprocess conf")
            preprocess_conf_file = self.config.collator.augmentation_config
            # redirect the cmvn path
            with io.open(preprocess_conf_file, encoding="utf-8") as f:
                preprocess_conf = yaml.safe_load(f)
                for idx, process in enumerate(preprocess_conf["process"]):
                    if process['type'] == "cmvn_json":
                        preprocess_conf["process"][idx][
                            "cmvn_path"] = os.path.join(
                                self.res_path,
                                preprocess_conf["process"][idx]["cmvn_path"])
                        break
            logger.info(preprocess_conf)
            preprocess_args = {"train": False}
            preprocessing = Transformation(preprocess_conf)
            logger.info("read the audio file")
            audio, audio_sample_rate = soundfile.read(
                audio_file, dtype="int16", always_2d=True)

            if self.change_format:
                if audio.shape[1] >= 2:
                    audio = audio.mean(axis=1, dtype=np.int16)
                else:
                    audio = audio[:, 0]
                # pcm16 -> pcm 32
                audio = self._pcm16to32(audio)
                audio = librosa.resample(audio, audio_sample_rate,
                                         self.sample_rate)
                audio_sample_rate = self.sample_rate
                # pcm32 -> pcm 16
                audio = self._pcm32to16(audio)
            else:
                audio = audio[:, 0]

            logger.info(f"audio shape: {audio.shape}")
            # fbank
            audio = preprocessing(audio, **preprocess_args)

            audio_len = paddle.to_tensor(audio.shape[0])
            audio = paddle.to_tensor(audio, dtype='float32').unsqueeze(axis=0)
            text_feature = TextFeaturizer(
                unit_type=self.config.collator.unit_type,
                vocab=self.config.collator.vocab_filepath,
                spm_model_prefix=self.config.collator.spm_model_prefix)
            self._inputs["audio"] = audio
            self._inputs["audio_len"] = audio_len
            logger.info(f"audio feat shape: {audio.shape}")

        else:
            raise Exception("wrong type")

    @paddle.no_grad()
    def infer(self, model_type: str):
        """
        Model inference and result stored in self.output.
        """
        text_feature = TextFeaturizer(
            unit_type=self.config.collator.unit_type,
            vocab=self.config.collator.vocab_filepath,
            spm_model_prefix=self.config.collator.spm_model_prefix)
        cfg = self.config.decoding
        audio = self._inputs["audio"]
        audio_len = self._inputs["audio_len"]
        if "ds2_online" in model_type or "ds2_offline" in model_type:
            result_transcripts = self.model.decode(
                audio,
                audio_len,
                text_feature.vocab_list,
                decoding_method=cfg.decoding_method,
                lang_model_path=cfg.lang_model_path,
                beam_alpha=cfg.alpha,
                beam_beta=cfg.beta,
                beam_size=cfg.beam_size,
                cutoff_prob=cfg.cutoff_prob,
                cutoff_top_n=cfg.cutoff_top_n,
                num_processes=cfg.num_proc_bsearch)
            self._outputs["result"] = result_transcripts[0]

        elif "conformer" in model_type or "transformer" in model_type:
            result_transcripts = self.model.decode(
                audio,
                audio_len,
                text_feature=text_feature,
                decoding_method=cfg.decoding_method,
                beam_size=cfg.beam_size,
                ctc_weight=cfg.ctc_weight,
                decoding_chunk_size=cfg.decoding_chunk_size,
                num_decoding_left_chunks=cfg.num_decoding_left_chunks,
                simulate_streaming=cfg.simulate_streaming)
            self._outputs["result"] = result_transcripts[0][0]
        else:
            raise Exception("invalid model name")

    def postprocess(self) -> Union[str, os.PathLike]:
        """
            Output postprocess and return human-readable results such as texts and audio files.
        """
        return self._outputs["result"]

    def _pcm16to32(self, audio):
        assert (audio.dtype == np.int16)
        audio = audio.astype("float32")
        bits = np.iinfo(np.int16).bits
        audio = audio / (2**(bits - 1))
        return audio

    def _pcm32to16(self, audio):
        assert (audio.dtype == np.float32)
        bits = np.iinfo(np.int16).bits
        audio = audio * (2**(bits - 1))
        audio = np.round(audio).astype("int16")
        return audio

    def _check(self, audio_file: str, sample_rate: int, force_yes: bool):
        self.sample_rate = sample_rate
        if self.sample_rate != 16000 and self.sample_rate != 8000:
            logger.error("please input --sr 8000 or --sr 16000")
            raise Exception("invalid sample rate")
            sys.exit(-1)

        if not os.path.isfile(audio_file):
            logger.error("Please input the right audio file path")
            sys.exit(-1)

        logger.info("checking the audio file format......")
        try:
            audio, audio_sample_rate = soundfile.read(
                audio_file, dtype="int16", always_2d=True)
        except Exception as e:
            logger.exception(e)
            logger.error(
                "can not open the audio file, please check the audio file format is 'wav'. \n \
                 you can try to use sox to change the file format.\n \
                 For example: \n \
                 sample rate: 16k \n \
                 sox input_audio.xx --rate 16k --bits 16 --channels 1 output_audio.wav \n \
                 sample rate: 8k \n \
                 sox input_audio.xx --rate 8k --bits 16 --channels 1 output_audio.wav \n \
                 ")
            sys.exit(-1)
        logger.info("The sample rate is %d" % audio_sample_rate)
        if audio_sample_rate != self.sample_rate:
            logger.warning("The sample rate of the input file is not {}.\n \
                            The program will resample the wav file to {}.\n \
                            If the result does not meet your expectations，\n \
                            Please input the 16k 16 bit 1 channel wav file. \
                        ".format(self.sample_rate, self.sample_rate))
            if force_yes is False:
                while (True):
                    logger.info(
                        "Whether to change the sample rate and the channel. Y: change the sample. N: exit the prgream."
                    )
                    content = "Y"   #input("Input(Y/N):")
                    if content.strip() == "Y" or content.strip(
                    ) == "y" or content.strip() == "yes" or content.strip(
                    ) == "Yes":
                        logger.info(
                            "change the sampele rate, channel to 16k and 1 channel"
                        )
                        break
                    elif content.strip() == "N" or content.strip(
                    ) == "n" or content.strip() == "no" or content.strip(
                    ) == "No":
                        logger.info("Exit the program")
                        exit(1)
                    else:
                        logger.warning("Not regular input, please input again")

            self.change_format = True
        else:
            logger.info("The audio file format is right")
            self.change_format = False

    
@singleton
class TextExecutor():
    def __init__(self):
        #super(TextExecutor, self).__init__()
        self._inputs = dict()
        self._outputs = dict()
        

    def _get_pretrained_path(self, tag: str) -> os.PathLike:
        """
            Download and returns pretrained resources path of current task.
        """
        assert tag in pretrained_models, 'Can not find pretrained resources of {}.'.format(
            tag)

        res_path = os.path.join(MODEL_HOME, tag)
        decompressed_path = download_and_decompress(pretrained_models[tag],
                                                    res_path)
        decompressed_path = os.path.abspath(decompressed_path)
        logger.info(
            'Use pretrained model stored in: {}'.format(decompressed_path))

        return decompressed_path

    def _init_from_path(self,
                        task: str='punc',
                        model_type: str='ernie_linear_p7_wudao',
                        lang: str='zh',
                        cfg_path: Optional[os.PathLike]=None,
                        ckpt_path: Optional[os.PathLike]=None,
                        vocab_file: Optional[os.PathLike]=None):
        """
            Init model and other resources from a specific path.
        """
        if hasattr(self, 'model'):
            logger.info('Model had been initialized.')
            return

        self.task = task

        if cfg_path is None or ckpt_path is None or vocab_file is None:
            tag = '-'.join([model_type, task, lang])
            self.res_path = self._get_pretrained_path(tag)
            self.cfg_path = os.path.join(self.res_path,
                                         pretrained_models[tag]['cfg_path'])
            self.ckpt_path = os.path.join(self.res_path,
                                          pretrained_models[tag]['ckpt_path'])
            self.vocab_file = os.path.join(self.res_path,
                                           pretrained_models[tag]['vocab_file'])
        else:
            self.cfg_path = os.path.abspath(cfg_path)
            self.ckpt_path = os.path.abspath(ckpt_path)
            self.vocab_file = os.path.abspath(vocab_file)

        model_name = model_type[:model_type.rindex('_')]
        if self.task == 'punc':
            # punc list
            self._punc_list = []
            with open(self.vocab_file, 'r') as f:
                for line in f:
                    self._punc_list.append(line.strip())

            # model
            model_class = dynamic_import(model_name, model_alias)
            tokenizer_class = dynamic_import(model_name, tokenizer_alias)
            self.model = model_class(
                cfg_path=self.cfg_path, ckpt_path=self.ckpt_path)
            # MODEL_HOME + "/" +   
            self.tokenizer = tokenizer_class.from_pretrained(MODEL_HOME + "/" + 'ernie-1.0')
        else:
            raise NotImplementedError

        self.model.eval()

    def _clean_text(self, text):
        text = text.lower()
        text = re.sub('[^A-Za-z0-9\u4e00-\u9fa5]', '', text)
        text = re.sub(f'[{"".join([p for p in self._punc_list][1:])}]', '',
                      text)
        return text

    def preprocess(self, text: Union[str, os.PathLike]):
        """
            Input preprocess and return paddle.Tensor stored in self.input.
            Input content can be a text(tts), a file(asr, cls) or a streaming(not supported yet).
        """
        if self.task == 'punc':
            clean_text = self._clean_text(text)
            assert len(clean_text) > 0, f'Invalid input string: {text}'

            tokenized_input = self.tokenizer(
                list(clean_text), return_length=True, is_split_into_words=True)

            self._inputs['input_ids'] = tokenized_input['input_ids']
            self._inputs['seg_ids'] = tokenized_input['token_type_ids']
            self._inputs['seq_len'] = tokenized_input['seq_len']
        else:
            raise NotImplementedError

    @paddle.no_grad()
    def infer(self):
        """
            Model inference and result stored in self.output.
        """
        if self.task == 'punc':
            input_ids = paddle.to_tensor(self._inputs['input_ids']).unsqueeze(0)
            seg_ids = paddle.to_tensor(self._inputs['seg_ids']).unsqueeze(0)
            logits, _ = self.model(input_ids, seg_ids)
            preds = paddle.argmax(logits, axis=-1).squeeze(0)

            self._outputs['preds'] = preds
        else:
            raise NotImplementedError

    def postprocess(self) -> Union[str, os.PathLike]:
        """
            Output postprocess and return human-readable results such as texts and audio files.
        """
        if self.task == 'punc':
            input_ids = self._inputs['input_ids']
            seq_len = self._inputs['seq_len']
            preds = self._outputs['preds']

            tokens = self.tokenizer.convert_ids_to_tokens(
                input_ids[1:seq_len - 1])
            labels = preds[1:seq_len - 1].tolist()
            assert len(tokens) == len(labels)

            text = ''
            for t, l in zip(tokens, labels):
                text += t
                if l != 0:  # Non punc.
                    text += self._punc_list[l]

            return text
        else:
            raise NotImplementedError

SAMPLING_RATE = 16000
FORCE_YES=True

def get_asr_text(audio_file):
    
    asr_executor = ASRExecutor()
    text_executor = TextExecutor()
    
    device=paddle.get_device()
    paddle.set_device(device)
    model = 'conformer_wenetspeech'
    asr_executor._init_from_path(model_type=model, lang='zh', sample_rate=SAMPLING_RATE, cfg_path=None, ckpt_path=None)
     
    text_executor._init_from_path(task='punc', model_type='ernie_linear_p3_wudao', lang='zh', cfg_path=None, ckpt_path=None, vocab_file=None)

    asr_executor._check(audio_file=audio_file, sample_rate=SAMPLING_RATE, force_yes=FORCE_YES)
    print("ASR CHECKED %s" % (datetime.now()))
    asr_executor.preprocess(model, audio_file)
    print("ASR PREPROCESSED %s" % (datetime.now()))
    asr_executor.infer(model)
    print("ASR INFERED %s" % (datetime.now()))
    text = asr_executor.postprocess()  # Retrieve result of asr.
    
    print("TEXT %s %s" % (datetime.now(), text))
    
    if len(text) > 0 and len(text) <= 512:
        text_executor.preprocess(text)
        text_executor.infer()
        text1 = text_executor.postprocess()
        print("TEXT INFERED %s" % (datetime.now()))
        return text1
    else:
        # 太长了的话，句读会报错
        return text
    
import torch,torchaudio
import json
import tempfile
import os

from utils_vad import init_jit_model,get_speech_timestamps, \
                     save_audio, \
                     read_audio, \
                     VADIterator, \
                     collect_chunks


@singleton
class VADExecutor():
    def __init__(self):
        self.model = init_jit_model(model_path=f'./files/silero_vad.jit')

    def seperate_segments(self, wav_file, sample_rate):
        wav = read_audio(path=wav_file, sampling_rate=sample_rate)
        # get speech timestamps from full audio file
        window_size_samples=768 if sample_rate==8000 else 1536
        speech_timestamps = get_speech_timestamps(wav, 
                                                self.model, 
                                                sampling_rate=sample_rate, 
                                                min_speech_duration_ms=1000, 
                                                min_silence_duration_ms=500, 
                                                window_size_samples=window_size_samples)
        return wav, speech_timestamps                                             

def gen_wav(wav_file, sample_rate=8000):
    vad_executor = VADExecutor()
    wav, speech_timestamps = vad_executor.seperate_segments(wav_file, sample_rate)

    # Temporary files are stored here
    temp_dir = "./tmp" #tempfile.gettempdir()
    for st in range(len(speech_timestamps)):
        tmp = os.path.join(temp_dir, "%s-%s.wav"%(wav_file, st))
    
        torchaudio.save(tmp,
                        collect_chunks([speech_timestamps[st]], wav).unsqueeze(0), 
                        sample_rate, 
                        encoding="PCM_S", 
                        bits_per_sample=16) 
        # 返回音频、以秒为单位的声音开始时间
        ts = speech_timestamps[st]['start']/sample_rate
        yield tmp, ts #str(timedelta(seconds=ts))

def get_wav_and_timestamps(wav_file, sample_rate=8000):
    vad_executor = VADExecutor()
    wav, speech_timestamps = vad_executor.seperate_segments(wav_file, sample_rate)
    temp_dir = "./tmp" #tempfile.gettempdir()
    wavs = []
    for st in range(len(speech_timestamps)):
        tmp = os.path.join(temp_dir, "%s.wav"%(st))
    
        torchaudio.save(tmp,
                        collect_chunks([speech_timestamps[st]], wav).unsqueeze(0), 
                        sample_rate, 
                        encoding="PCM_S", 
                        bits_per_sample=16) 
        wavs.append(tmp)
    return wav, speech_timestamps

if __name__ == '__main__':
    import sys
    if len(sys.argv) == 3:
        wav_file = sys.argv[1]
        txt_file = sys.argv[2]
    elif len(sys.argv) == 2:
        wav_file = sys.argv[1]
        txt_file = 'asr_output.txt'
    else:
        wav_file = './audioData.wav'
        txt_file = 'asr_output.txt'
    text=""
    with open(txt_file, "a+") as f:
        for t, ts in gen_wav(wav_file):
            print(t, ts)
            print("INITED %s" % (datetime.now()))
            asrt=get_asr_text(t)
            f.write("%s: %s\r\n" % (ts, asrt))
            f.flush()
            text = text + asrt
            #print( text )              
        