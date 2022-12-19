import matplotlib.pyplot as plt
import seaborn as sns

from tqdm import tqdm
tqdm.pandas()

def utils_compute_compound_score(analyzer, tweet):
    """
    Compute the compound score for a given tweet using a SentimentIntensityAnalyzer object.
    
    Parameters:
        analyzer (SentimentIntensityAnalyzer): The SentimentIntensityAnalyzer object to use.
        tweet (str): The tweet to analyze.
    
    Returns:
        float: The compound score for the tweet.
    """
    return analyzer.polarity_scores(tweet)['compound']


def utils_get_text_features(dataframe, text_column):
    # Add a column to the dataframe with the word count for each text
    dataframe['word_count'] = dataframe[text_column].progress_apply(lambda x: len(str(x).split(" ")))
    
    # Add a column to the dataframe with the character count for each text
    dataframe['char_count'] = dataframe[text_column].progress_apply(lambda x: sum(len(word) for word in str(x).split(" ")))
    
    # Add a column to the dataframe with the average word length for each text
    dataframe['avg_word_length'] = dataframe['char_count'] / dataframe['word_count']
    
    # Add a column to the dataframe with the sentence count for each text
    dataframe['sentence_count'] = dataframe[text_column].apply(lambda x: len(str(x).split(".")))
    
    # Add a column to the dataframe with the average sentence length for each text
    dataframe['avg_sentence_length'] = dataframe['word_count'] / dataframe['sentence_count']


def utils_plot_hist_and_density(dataframe, x, y):
    # Create a figure with 1 row and 2 columns
    fig, ax = plt.subplots(nrows=1, ncols=2)
    
    # Set the title for the figure
    fig.suptitle(x, fontsize=12)
    
    # Iterate over the unique values in the y column
    for i in dataframe[y].unique():
        # Plot a histogram for the x column, grouped by the current value of y
        sns.distplot(dataframe[dataframe[y]==i][x], hist=True, kde=False, 
                     bins=10, hist_kws={"alpha":0.8}, 
                     axlabel="histogram", ax=ax[0])
        
        # Plot a density plot for the x column, grouped by the current value of y
        sns.distplot(dataframe[dataframe[y]==i][x], hist=False, kde=True, 
                     kde_kws={"shade":True}, axlabel="density",   
                     ax=ax[1])
    
    # Show grid lines on the histogram plot
    ax[0].grid(True)
    
    # Set the legend for the histogram plot
    ax[0].legend(dataframe[y].unique())
    
    # Show grid lines on the density plot
    ax[1].grid(True)
    
    # Display the plots
    plt.show()


def utils_value_counts_and_pct_change(dataframe):
    # Calculate the value counts of the 'label' column
    counts = dataframe['label'].value_counts()

    # Calculate the maximum percentage change of the value counts
    max_pct_change = counts.pct_change().max()

    # Print the value counts and the maximum percentage change
    print(counts)
    print(f"Percentage Difference: {round(max_pct_change, 5)}")

    # Plot a countplot using the 'label' column
    sns.countplot(dataframe.label)


def remove_empty_rows(dataframe, column):
    # Remove empty text rows
    df = dataframe[dataframe[column].str.len() > 0]

    # Reset index
    df = df.reset_index(drop=True)
    return df