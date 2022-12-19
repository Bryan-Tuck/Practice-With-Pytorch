import matplotlib.pyplot as plt
import seaborn as sns

from tqdm import tqdm
tqdm.pandas()

def utils_get_text_features(df, text_column):
    # Add a column to the dataframe with the word count for each text
    df['word_count'] = df[text_column].progress_apply(lambda x: len(str(x).split(" ")))
    
    # Add a column to the dataframe with the character count for each text
    df['char_count'] = df[text_column].progress_apply(lambda x: sum(len(word) for word in str(x).split(" ")))
    
    # Add a column to the dataframe with the average word length for each text
    df['avg_word_length'] = df['char_count'] / df['word_count']
    
    # Add a column to the dataframe with the sentence count for each text
    df['sentence_count'] = df[text_column].apply(lambda x: len(str(x).split(".")))
    
    # Add a column to the dataframe with the average sentence length for each text
    df['avg_sentence_length'] = df['word_count'] / df['sentence_count']


def utils_plot_hist_and_density(df, x, y):
    # Create a figure with 1 row and 2 columns
    fig, ax = plt.subplots(nrows=1, ncols=2)
    
    # Set the title for the figure
    fig.suptitle(x, fontsize=12)
    
    # Iterate over the unique values in the y column
    for i in df[y].unique():
        # Plot a histogram for the x column, grouped by the current value of y
        sns.distplot(df[df[y]==i][x], hist=True, kde=False, 
                     bins=10, hist_kws={"alpha":0.8}, 
                     axlabel="histogram", ax=ax[0])
        
        # Plot a density plot for the x column, grouped by the current value of y
        sns.distplot(df[df[y]==i][x], hist=False, kde=True, 
                     kde_kws={"shade":True}, axlabel="density",   
                     ax=ax[1])
    
    # Show grid lines on the histogram plot
    ax[0].grid(True)
    
    # Set the legend for the histogram plot
    ax[0].legend(df[y].unique())
    
    # Show grid lines on the density plot
    ax[1].grid(True)
    
    # Display the plots
    plt.show()