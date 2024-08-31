import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from sklearn.utils import shuffle
import nlpaug.augmenter.word.context_word_embs as aug

class DataAugmentor:
    def __init__(self, model_path='bert-base-uncased', action="insert"):
        self.augmenter = aug.ContextualWordEmbsAug(model_path=model_path, action=action)

    def augment_data(self, df, repetitions=1, samples=1000):
        augmented_texts = []
        # Select only the minority class samples
        minority_df = df[df['sentiment'] == 0].reset_index(drop=True)
        
        for i in tqdm(np.random.randint(0, len(minority_df), samples)):
            # Generating 'samples' number of augmented texts
            for _ in range(repetitions):
                augmented_text = self.augmenter.augment(minority_df['cleaned_content'].iloc[i])
                augmented_texts.append(augmented_text)

        data = {
            'sentiment': 0,
            'cleaned_content': augmented_texts
        }
        aug_df = pd.DataFrame(data)
        df = shuffle(pd.concat([df, aug_df]), random_state=42).reset_index(drop=True)
        return df