import matplotlib.pyplot as plt
import numpy as np
import hashlib
import os
from random import shuffle
from multiprocessing import Pool
import multiprocessing
import traceback
from time import sleep
from random import randint

from modules.logging import logger
import modules.logging as logging
import modules.utils as utils
from modules.utils import Timer

class ShardGroup:
    
    def __init__(self, items, nr_shards, base_shards_dir, recreate_shards_dir=False, random_seed=0.1):
        """
            items: input items that will be separated into shard groups. for shard separation this item will be converted to string using 'str' and than its hash will be used for group separation
            base_shards_dir: directory where an output folder will be created for this shard
            random_seed: random seed used during shuffling
            nr_shards: number of shards to separate groups of items
        """
        self.nr_shards = nr_shards
        self.items = items
        self.base_shards_dir = base_shards_dir
        self.random_seed = random_seed
        utils.mkdirs(base_shards_dir, dirs=[], recreate=recreate_shards_dir)
        #logging.setup_file_logger(base_shards_dir + 'out.log')
        
    def shard_items(self, shard_id):
        """
            Select some items for the specified shard_id. Returned items will be different from one shard to another.
            shard_id: 1-N shard number
            returns: list of items for this shard
        """
        shard_items = []

        for item in self.items:
            p = hashlib.sha224(str(item).encode('utf-8')).hexdigest()
            if(int(p,16)%self.nr_shards == (shard_id-1)):
                shard_items.append(item)
        logger.info('found {} items for shard {}'.format(len(shard_items), shard_id))
        shuffle(shard_items, lambda: self.random_seed)
        return shard_items

    def shard_dir(self, shard_id, shard_dirs=[]):
        """
            Select some items and create a output dir for this shard
            shard_id: 1-N shard number
            recreated_dir: delete and recreate shard output dir
            returns: shard_items, shard_dir, shard_done
                     or: list of items for this shard, shard directory created for this shard results, True if file 'done' already exists in shard dir
        """
        shard_dir = self.base_shards_dir + str(shard_id) + '/'
        utils.mkdirs(shard_dir, dirs=shard_dirs, recreate=False)
        return shard_dir

    def shard_path(self, shard_id):
        return self.base_shards_dir + str(shard_id) + '/'
    
    def shard_done(self, shard_id):
        #check if this shard was already processed
        file_done = self.shard_path(shard_id) + 'done'
        return os.path.isfile(file_done)
 
    def mark_done(self, shard_id, info='OK'):
        file_done = self.shard_path(shard_id) + 'done'
        f = open(file_done, 'w')
        f.write(info)
        f.close()

    def shard_dirs(self):
        return [self.shard_dir(sid) for sid in range(1,self.nr_shards+1)]
            
    def start_processing(self, process_shard_function, threads=-1, ramp_delay=(1,10), shard_ids=None):
        """
            Start processing shards in parallel using Threads. 
            process_shard_function: must support parameters 'shard_group' and 'shard_id'
            threads: if -1, will use all available cores
            ramp_sleep: random time in seconds to wait between Thread launches in format (min, max) seconds.
            shard_ids: shard_ids to be processed. If None, all shards will be processed
        """
        
        #mp.set_start_method('spawn')
        if(threads<0):
            threads = multiprocessing.cpu_count()
        logger.info('Using ' + str(threads) + ' parallel tasks')

        with Pool(threads) as p:
            if(shard_ids==None):
                shard_ids = list(range(1,self.nr_shards+1))
            shuffle(shard_ids)
            p.starmap(self.process_shard, [(sid,process_shard_function,ramp_delay) for sid in shard_ids])

    def process_shard(self, shard_id, process_shard_function, ramp_delay):
        try:
            sleep(randint(ramp_delay[0],ramp_delay[1]))
            return process_shard_function(self, shard_id)
        except BaseException as e:
            details = traceback.format_exc()
            if(details==None):
                details = 'None'
            logger.warning('Exception while processing shard ' + str(shard_id) + ': ' + str(e) + '; details=' + details)
            return 'shard ' + str(shard_id) + ' exception: ' + str(e)
