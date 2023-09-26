from pytorch_lightning.callbacks import Callback
import numpy as np

class ExportSplitCallback(Callback):
    def on_train_start(self, trainer, pl_module):
        
        if trainer.datamodule.__class__.__name__ == 'HSIRoad':
            train_samples = np.array(trainer.datamodule.dataset_train.name_list)
            val_samples = np.array(trainer.datamodule.dataset_val.name_list)
            # is unfortunately the same as validation
            test_samples = np.array(trainer.datamodule.dataset_val.name_list)
        else:

            # access full list of sample-names through
            # trainer -> datamoudle -> train subset object -> original dataset object
            sample_list = np.array(trainer.datamodule.dataset_train.dataset.samplelist())

            # get list of indices of train, test and validation subsets
            train_samples = np.array(sample_list[trainer.datamodule.dataset_train.indices])
            val_samples = np.array(sample_list[trainer.datamodule.dataset_val.indices])
            test_samples = np.array(sample_list[trainer.datamodule.dataset_test.indices])
        
        # store list of train, validation and test sample names in log-dir
        np.savetxt(trainer.logger.log_dir+'/train_samples.csv', train_samples, fmt='%s')
        np.savetxt(trainer.logger.log_dir+'/val_samples.csv', val_samples, fmt='%s')
        np.savetxt(trainer.logger.log_dir+'/test_samples.csv', test_samples, fmt='%s')
