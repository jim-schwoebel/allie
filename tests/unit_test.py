'''
Simple unit test the default parameters 
of the repository. 

Note this unit_testing requires the settings.json
to defined in the base directory.
'''
import unittest

class SimplisticTest(unittest.TestCase):
     
    def test(self):
        a = 'a'
        b = 'a'
        self.assertEqual(a, b)
    
    # confirm that all the modules are installed correctly 
    # {} - list of modules imported requirements.txt 
          
    # can read audio files
    # can read text files 
    # can read image files 
    # can read video files 
     
    # can transcribe audio files 
    # can transcribe image files 
    # can transcribe video files 

    # can fetch dataset (common_voice) in folder (internet connection exists) 
     
    # can augment dataset - audio files (audio_augmentation) 
    # can augment dataset - text files (text_augmentation) 
    # can augment dataset - image files (image_augmentation)
    # can augment dataset - video files (video_augmentation) 

    # can featurize audio files via specified featurizer (can be all featurizers) 
    # can featurize text files via specified featurizer (can be all featurizers)
    # can featurize image files via specified featurizer (can be all featurizers) 
    # can featurize video files via specified featurizer (can be all featurizers) 
     
    # can train machine learning model via specified trainer (test)

    # can load SC machine learning models
    # can load TPOT models 
    # can load keras deep learning models 
    # can load ludwig deep learning models 

    # if applicable, can compress audio models 
     
    # if applicable, can create YAML files (for production) 
     
if __name__ == '__main__':
    unittest.main()
