import re
import numpy as np
import pandas as pd
import random

from bs4 import BeautifulSoup
import urllib.request

import matplotlib.pyplot as plt
import seaborn as sns

from textblob import TextBlob
import time

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

from nltk.corpus import stopwords
from wordcloud import WordCloud

def pct_func(pct, raw):
    """
    Function:
    Modify the annotation of percent in pie plot.
    """
    absolute = int(round(pct/100.*np.sum(raw)))
    return("{:d}\n({:.1f}%)".format(absolute, pct))

def get_my_palette():
    """
    Function:
    Generate two palette of colors for each media type and each language (including 33 most popular languages).
    """
    rclrset1 = ["#E41A1C", "#377EB8", "#4DAF4A", "#984EA3", "#FF7F00", "#FFFF33", "#A65628", "#F781BF", "#999999"]
    rclrset2 = ["#66C2A5", "#FC8D62", "#8DA0CB", "#E78AC3", "#A6D854", "#FFD92F", "#E5C494", "#B3B3B3"]
    rclrset3 = ["#8DD3C7", "#FFFFB3", "#BEBADA", "#FB8072", "#80B1D3", "#FDB462", "#B3DE69", "#FCCDE5", "#D9D9D9", "#BC80BD", "#CCEBC5", "#FFED6F"]
    rclrdark2 = ["#1B9E77", "#D95F02", "#7570B3","#E7298A", "#66A61E", "#E6AB02", "#A6761D", "#666666"]
    
    lang_color = {}
    color33 = rclrset1 + rclrset2 + rclrset3 + rclrdark2
    random.seed(1)
    random.shuffle(color33)    
    langs = ['ar', 'ca', 'co', 'cs', 'da', 'de', 'el', 'en', 'es', 'fi', 'fr', 'hi', 'ht', 'hu', 'id', 'it', 
             'ja', 'ko', 'la', 'lb', 'mi', 'mr', 'nl', 'no', 'pl', 'pt', 'ro', 'ru', 'sv', 'th', 'tr','zh-CN','zh-TW']
    for i,lang in enumerate(langs):
        lang_color[lang] = color33[i]
    
    media_color = {}
    suffix = ['Anime & Manga', 'Books & Literature', 'Cartoons & Comics & Graphic Novels', 
          'Movies', 'Music & Bands', 'Other Media', 'Celebrities & Real People', 
          'Theater', 'TV Shows', 'Video Games']
    for i,m in enumerate(suffix):
        media_color[m] = (rclrdark2+rclrset2)[i]
    
    return(media_color, lang_color)

def get_stop_words():
    """
    Function:
    Map language code to NLTK's stop words.
    """
    stop_lang_dict = {"fr": "french", "de": "german", "es": "spanish", "it": "italian", 
                      "nl": "dutch", "sv": "swedish","no": "norwegian", "en": "english"
                     }
    return(stop_lang_dict)

def get_sns_palette(name_list, existing_palette):
    """
    Function:
    Create/supplement existing palette.
    Input:
    - name_list: List[str], name of color
    - existing_palette: dict, existing palette
    Output:
    - existing_palette: dict
    """
    snspalette = sns.color_palette(None, len(name_list))
    
    for i,n in enumerate(name_list):
        if n not in existing_palette:
            existing_palette[n] = snspalette[i]
    
    return(existing_palette)

def CrawlTags(mediaTypes):
    """
    Function: 
    Crawl fandom tags for each media type from ao3.
    
    Input:
    - mediaTypes: List[str]. Each str is a media type.
    
    Output:
    - df: pandas.DataFrame. Columns include:
      - fantom: str
      - cnt: int
      - href: str
      - media_type: str
    """
    dflist = []
    
    for mt in mediaTypes:
        mtstr = mt.replace("&","*a*").replace(" ","%20")
        baseurl = "https://archiveofourown.org/media/" + mtstr + "/fandoms"
        content = urllib.request.urlopen(baseurl)
        soup = BeautifulSoup(content, "html.parser")
        ul_all = soup.findAll("ul", attrs = {"class": "tags index group"})
        
        idx = 0
        mydict = {}
        for ul in ul_all:
            lis = ul.findAll("li")
            for li in lis:
                textls = li.text.split("\n")
                textls = [tx.strip() for tx in textls if len(tx.strip()) > 0]
                mydict[idx] = {"fantom": textls[0], \
                               "cnt": int(textls[1].rstrip(")").lstrip("(")), \
                               "href":li.find("a")["href"]
                               }
                idx += 1

        dffantom = pd.DataFrame.from_dict(mydict, orient = "index")
        dffantom["MediaType"] = mt
        dflist.append(dffantom)
        time.sleep(2)
    
    df = pd.concat(dflist, ignore_index=True)
    df['index'] = df.index
    return(df)

def AnnotateLanguage(df):
    """
    Function:
    Detect the language of each tag based on the name of the work and the author.
    - Format of `fantom`:
        - if English: `work - author` (e.g. `Harry Potter - J. K. Rowling`);
        - otherwise: translation seprated by '|' (e.g. La Belle au bois dormant | Sleeping Beauty - Charles Perrault).
    - Languages are detected by calling TextBlob:
        - TextBlob uses the Google Translate API and requires an internet connection.
        - To avoid HTTP 429 Too Many Requests Error with the following command:
            >>> dffantom['org_l = dffantom['org_ftm'].apply(lambda x: TextBlob(x).detect_language())']
          use iterrows instead (it takes some time).
    
    Input:
    - df: pandas.DataFrame.
    
    Output:
    - df: pandas.DataFrame.
    """
    
    df["org_ftm"] = df["fantom"].apply(lambda x: x.split("|")[0])
    df["eng_ftm"] = df["fantom"].apply(lambda x: x.split("|")[-1])
    df["org_lang"] = "unknown"
    
    for i,row in df.iterrows():
        txt = row["org_ftm"]
        if len(txt) >= 3:
            df.at[i,"org_lang"] = TextBlob(txt).detect_language()
        time.sleep(0.5)
    
    df['isEn'] = np.where(df['org_lang'] == "en", "en", "not en")
    return(df)

def CorrectLanguage(df):
    """
    Function:
    Manually correct languages that are clearly mistakenly labeled (based on YD's limited world knowledge).
    
    Input:
    - df: pandas.DataFrame
    
    Output:
    - df: pandas.DataFrame
    """
    lang_corrections = {'Harry Potter':'en', 
                        'Jane Austen':'en',
                        'Naruto':'ja', 
                        'Yuri!!! on Ice':'ja',
                        'Hetalia: Axis Powers':'ja',
                        'One Piece':'ja', 
                        'Fairy Tail':'ja', 
                        'Tennis no Oujisama':'ja',
                        'Tenshi Kinryoku':'ja', 
                        '南派三叔':'zh-CN',
                        'Katanagatari': 'ja',
                        'ACCA13区監察課':'ja'
                       }

    for f in lang_corrections:
        df.at[df['fantom'].str.contains(f), 'org_lang'] = lang_corrections[f]
    return(df)

def Vis_TagCntPerMedia(df):
    """
    Function:
    Visualize count of tags of each media in a line plot.
    
    Input:
    - df: pandas.DataFrame.
    
    Output:
    - dfm: pandas.DataFrame, count of tags for each media.
    - fig: matplotlib.figure.Figure.
    """
    dfm = df[['index','MediaType']].groupby(['MediaType']).count().reset_index().rename(columns = {'index':'count'}).sort_values(by = ['count'],ascending=False)
    
    fig, ax = plt.subplots()
    ax.plot(dfm['MediaType'], dfm['count'], 'o-', color = '#6A5ACD')
    plt.xlabel("Media Type", fontsize = 14, fontweight="bold")
    plt.ylabel("count", fontsize = 14, fontweight="bold")
    plt.xticks(fontsize=12, rotation = 90)
    plt.yticks(fontsize=12)
    plt.title("# Tags by Media Type", fontsize = 14, fontweight="bold")
    plt.grid(True)
    ax.set_ylim([min(dfm['count'])*0.9, max(dfm['count'])*1.1])

    for i in range(dfm.shape[0]):
        plt.annotate(str(dfm.loc[i]['count']), 
                     (dfm.loc[i]['MediaType'], dfm.loc[i]['count']*1.02), 
                     fontsize = 10)
    
    return(dfm, fig)

def Vis_TagCntPerLang(df, min_cnt = 100):
    """
    Function:
    Visualize count of tags of each language, in two pie plots.
    
    Input:
    - df: pandas.DataFrame.
    - min_cnt: int, to include only languages having tags >= min_cnt.
    
    Output:
    - dfl_freq: pandas.DataFrame, count of tags for each language.
    - fig: matplotlib.figure.Figure.
    """
    df_lang = df[['index','org_lang','isEn']].groupby(['org_lang','isEn']).count().reset_index().rename(columns = {'index':'count'}).sort_values(by = ['count'],ascending=False)
    df_lang['rank'] = range(1, df_lang.shape[0]+1)
    dfl_freq = df_lang.loc[df_lang['count'] >= min_cnt].reset_index()
    
    dfl_freq2 = dfl_freq[['count',"isEn"]].groupby("isEn").sum().reset_index()
    dfl_notEn = dfl_freq.loc[dfl_freq['isEn'] == "not en"]

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 15))

    wedges, _ , autotexts = axes[0].pie(x = dfl_freq2['count'], 
           explode = [0, 0.1],
           labels = dfl_freq2['isEn'], 
           autopct = lambda pct: pct_func(pct, dfl_freq2['count']),
           shadow = True, startangle = 90,
           textprops = {"fontsize": 14})
    en_palette = {"en":"#bfefff", "not en": "#FFE4BF"}
    for i,wdg in enumerate(wedges):
            wdg.set_color(en_palette[dfl_freq2.loc[i]['isEn']])            
    axes[0].set_title("Proportion of `en` vs. `not en`", fontsize = 14, fontweight="bold")

    _, _, autotexts = axes[1].pie(x = dfl_notEn['count'], 
                explode = [0.0]* (dfl_notEn.shape[0]),
           labels = dfl_notEn['org_lang'], 
           autopct='%1.0f%%',
           startangle = 90,
           pctdistance = 0.8,
           textprops = {"fontsize": 10})
    for autotext in autotexts:
        autotext.set_color('white')
    axes[1].set_title("Proportion of languages in `not en`", fontsize = 14, fontweight = "bold")
    
    return(dfl_freq, fig)

def GroupByLangMedia(df):
    """
    Function:
    Group by both language and media type.
    """
    df_media_lang = df[['index','org_lang','MediaType','isEn']].groupby(['MediaType','org_lang','isEn']).count().reset_index().rename(columns = {'index':'count'}).sort_values(by = ['count'],ascending=False)
    return(df_media_lang)

def Vis_LangByMedia(df, ncol = 4, quantile = 1, lang_color = {}):
    """
    Function:
    Visualize proportions of tags in each language for every media type.
    Input:
    - df: pandas.DataFrame.
    - ncol: int, number of pie plots per row.
    - quantile: float in (0,1], the quantile of languages to include (control for # of languages to plot).
    - lang_color: dict, a palette mapping language to color (e.g. {"en": "red"}).
    Ouput:
    - dfml: pandas.DataFrame, count of tags for each media and each language.
    - fig: matplotlib.figure.Figure.
    """
    dfml = GroupByLangMedia(df)
    
    medias = list(set(list(dfml['MediaType'])))
    medias.sort()
    nrow = len(medias) // ncol + (len(medias) % ncol > 0)
    
    lang_color = get_sns_palette(list(set(list(df['org_lang']))), lang_color)

    # plot
    fig1, axes1 = plt.subplots(nrows = nrow, ncols = ncol, figsize = (ncol*4, nrow*4))
    fig2, axes2 = plt.subplots(nrows = nrow, ncols = ncol, figsize = (ncol*4, nrow*4))
    
    if nrow <= 1:
        print("Please provide a new value of ncol that is smaller than or equal to %d" % (len(medias)-1))
        return(dfml, fig1, fig2)

    for imedias, s in enumerate(medias):
        dftmp = dfml.loc[dfml['MediaType'] == s]
        dftmp.loc[:,'rank'] = range(1, dftmp.shape[0]+1)
        dftmp.loc[:,'cumprop'] = dftmp['count'].cumsum()/np.sum(dftmp['count'])

        ind = min(dftmp.loc[dftmp['cumprop'] >= quantile]['rank'])
        dfpop = dftmp.loc[dftmp['rank'] <= ind]

        dfpopEn = dfpop[['count',"isEn"]].groupby("isEn").sum().reset_index()
        dfpopnotEn = dfpop.loc[dfpop['isEn'] == "not en"].reset_index()
        
        # plot 1: en vs. not en
        rid, cid = imedias // ncol, imedias % ncol
        wedges, _ , autotexts = axes1[rid][cid].pie(x = dfpopEn['count'], 
                            explode = [0] + [0.1]*(dfpopEn.shape[0]-1),
                                      labels = dfpopEn['isEn'],
                                      autopct = lambda pct: pct_func(pct, dfpopEn['count']),
                                      startangle = 90,
                                      colors = ['#bfefff','#FFE4BF'],
                                      textprops = {"fontsize": 14})
        en_palette = {"en":"#bfefff", "not en": "#FFE4BF"}
        for i,wdg in enumerate(wedges):
                wdg.set_color(en_palette[dfpopEn.loc[i]['isEn']]) 
        axes1[rid][cid].set_title("`en` vs. `not en`: \n" + s, fontsize = 14, fontweight="bold")

        # plot 2: org_lang as in `not en`
        wedges, _ , autotexts = axes2[rid][cid].pie(x = dfpopnotEn['count'], 
                                      labels = dfpopnotEn['org_lang'],
                                      autopct='%1.0f%%', 
                                      startangle = 90,
                                      pctdistance = 0.8,
                                      textprops = {"fontsize": 10})
        
        for i,wdg in enumerate(wedges):
            wdg.set_color(lang_color[dfpopnotEn.loc[i]['org_lang']])

        for autotext in autotexts:
            autotext.set_color('white')

        axes2[rid][cid].set_title("Proportion of `not en`: \n" + s, 
                                                         fontsize = 14, 
                                                         fontweight="bold")
        
    # delete empty figs
    for j in range(imedias + 1, nrow*ncol):
        fig1.delaxes(axes1[j // ncol][j % ncol])
        fig2.delaxes(axes2[j // ncol][j % ncol])

    return(dfml, fig1, fig2)

def Vis_MediaByLang(df, langs, 
                    do_cluster = False, 
                    max_ncol = 5, 
                    ncluster = 4,
                    media_color = {}):
    """
    Function:
    Visualize proportions of tags in each language for every media type.
    Input:
    - df: pandas.DataFrame.
    - langs: list[str], languages of interest
    - do_cluster: Boolen, if True, starts a new row in the plot for a new cluster of languages that behave similarly regarding #tag distribution per media.
    - max_ncol: int, maximum plots per row.
    - media_color: dict, a palette mapping media to color.
    Ouput:
    - dfml2: pandas.DataFrame, count of tags for each media and each language. If do_cluster = True, a column indicating clusters is added.
    - fig: matplotlib.figure.Figure.
    - fig_cluster: a plot of language clusters if do_cluster = True, a None object otherwise.
    """
    dfml = GroupByLangMedia(df)
    dfml2 = dfml.loc[dfml['org_lang'].isin(langs)].reset_index(drop=True)
    
    # location of subplots
    langset = list(set(list(dfml2['org_lang'])))
    ncol = min(max_ncol, len(langset)-1)

    if do_cluster == True:
        pcadf, fig_cluster, _ = ClusterLang(dfml2, ncluster)     
        classdict = {i:row['class'] for i,row in pcadf.iterrows()}
        dfml2['class'] = dfml2['org_lang'].apply(lambda x: classdict[x])
        
        cluster_clr = get_sns_palette(set(list(pcadf['class'])), {})
    else:
        classdict = {j:i//ncol for i,j in enumerate(langset)}
        fig_cluster = None
        
        cluster_clr = {classdict[k]:"black" for k in classdict}
        
    classdict_rev = {}
    for k in classdict.keys():
        classdict_rev.setdefault(classdict[k], []).append(k)

    class_ls0 = [(len(classdict_rev[k]),k) for k in classdict_rev.keys()]
    class_ls0.sort()
    class_ls0.reverse()
    # add fillers in case size of cluster > ncol
    class_ls = sum([[i[1]]+[None]*(i[0]//ncol - 1 + (i[0] % ncol > 0)) for i in class_ls0],[])
    
    media_color = get_sns_palette(list(set(list(df['MediaType']))), media_color)
    nrow = len(class_ls)
    
    # plot
    fig, axes = plt.subplots(nrows = nrow, ncols = ncol, figsize = (ncol*5, nrow*5))
    all_axes = [(i,j) for i in range(nrow) for j in range(ncol)]
    ploted_axes = []
    if nrow <= 1:
        print("Please provide a new value of max_ncol that is smaller than or equal to %d" % (len(mlangs)-1))
        return(dfml2, fig)
    
    for il,lang in enumerate(langset):
        dfpop = dfml2.loc[dfml2['org_lang'] == lang].reset_index()

        class_lang = classdict[lang]
        class_lang_id = classdict_rev[class_lang].index(lang)
        class_id = class_ls.index(class_lang) + (class_lang_id // ncol)

        wedges, _ , autotexts = axes[class_id][class_lang_id % ncol].pie(x = dfpop['count'], 
                                      labels = dfpop['MediaType'],
                                      autopct='%1.0f%%', 
                                      startangle = 90,
                                      pctdistance = 0.8,
                                      textprops={"fontsize": 10})
        
        for i,wdg in enumerate(wedges):
            wdg.set_color(media_color[dfpop.loc[i]['MediaType']])

        for autotext in autotexts:
            autotext.set_color('white')

        axes[class_id][class_lang_id % ncol].set_title(lang, fontsize = 14, 
                                                       fontweight="bold", color = cluster_clr[class_lang])
        ploted_axes.append((class_id, class_lang_id % ncol))
    
    # delete empty subplots
    for a in all_axes:
        if a not in ploted_axes:
            ar,ac = a
            fig.delaxes(axes[ar][ac])
    
    return(dfml2, fig, fig_cluster)

def ClusterLang(df, ncluster):
    """
    Function:
    Cluster languages that behave similarly in terms of the distribution of tags across medias.
    Input:
    - df: pandas.DataFrame.
    - ncluster: int, number of clusters for KMeans.
    Output:
    - pcadf: pandas.DataFrame, results of PCA (including 2-D coordinates and class of kmeans clustering).
    - ax: scatter plot of clusters.
    - pca_explained_var: variance explained by each of PCA's components.
    """
    dfml3 = df.set_index(['org_lang','MediaType'])['count'].unstack(fill_value = 0)
    dfml3 = dfml3.div(dfml3.sum(axis = 1), axis = 0)

    X = dfml3.loc[:, dfml3.columns != 'other'].values
    pca = PCA(n_components = 2)
    pca.fit(X)

    pca_explained_var = pca.explained_variance_ratio_

    X2 = pca.fit_transform(X)
    pcadf = pd.DataFrame(X2)
    pcadf.columns = ['x1','x2']
    pcadf.index = dfml3.index

    pcadf['class'] = KMeans(n_clusters = ncluster).fit_predict(pcadf.values)

    # plot
    pcadf['class'] = pcadf['class'].apply(lambda x: chr(x + 65))
    ax = sns.scatterplot(x = "x1", y = "x2", hue = "class", data = pcadf)
    ax.legend(loc = 'center left', bbox_to_anchor = (1.25, 0.5), ncol = 1)
    for i, row in pcadf.iterrows():
        ax.text(row['x1'] + .02, row['x2'], i)
    
    return(pcadf, ax, pca_explained_var)

def Vis_PopularTags(df, top_k = 5, one_media_only = True, media_color = {}):
    """
    Function:
    Visualize top k tags that yield most works in each media.
    Input:
    - df: pandas.DataFrame.
    - top_k: int, number of top tags to include.
    - one_media_only: Boolean, tags with 'All Media Types' are excluded if True.
    - media_color: dict, palette for media types.
    Output:
    - dftop: pandas.DataFrame.
    - g: bar plot.
    """
    df_m_one= df.loc[~df['fantom'].str.contains("- All Media Types")]
    df_m_one_top = df_m_one.groupby("MediaType")['fantom','cnt','MediaType'].apply(lambda x: x.nlargest(top_k,['cnt'])).reset_index(drop=True)
    df_m_one_top['isAllMedia'] = False

    df_m_multi = df.loc[df['fantom'].str.contains("- All Media Types")]
    df_m_multi_top = df_m_multi.groupby("MediaType")['fantom','cnt','MediaType'].apply(lambda x: x.nlargest(top_k,['cnt'])).reset_index(drop=True)
    df_m_multi_top['isAllMedia'] = True
    
    if one_media_only:
        dftop = df_m_one_top
    else:
        dftop = pd.concat([df_m_one_top,df_m_multi_top], ignore_index=True)
    
    dftop['rank'] = dftop.sort_values(by=['cnt'], ascending=False).groupby("MediaType").cumcount() + 1
    dftop['fantom_short'] = dftop['fantom'].apply(lambda x: x.split("|")[-1].split(" - ")[0])
    
    # color palette
    AllMediaColor = {True: "red", False: "black"}
    medias = list(set(list(dftop['MediaType'])))
    media_color = get_sns_palette(medias, media_color)
            
    # plot
    g = sns.FacetGrid(dftop, 
                      col="MediaType", 
                      hue="MediaType", 
                      col_wrap = 3, 
                      sharex = False,
                      sharey = False, 
                      palette = media_color)
    g = (g.map(sns.barplot, "rank", "cnt", edgecolor="w").add_legend())
    g.fig.set_size_inches(15,15)

    for ax in g.axes:
        # set y label
        ylabels = ['{:,.0f}'.format(x) + 'K' for x in ax.get_yticks()/1000]
        ax.set_yticklabels(ylabels)

        # annotate bars
        media = ax.get_title().replace("MediaType = ","")
        yval = np.max(dftop.loc[(dftop['MediaType'] == media)]['cnt'])

        for x in ax.get_xticklabels():
            xval = x.get_text()

            tmp = dftop.loc[(dftop['MediaType'] == media) & (dftop['rank'] == int(xval))]
            if tmp.shape[0] > 0:
                xftm = str(tmp['fantom_short'].values[0])
                ax.annotate(xftm, 
                            xy = (int(xval)-1.1, yval), 
                            rotation = 90, 
                            color = AllMediaColor[tmp['isAllMedia'].values[0]])
    
    return(dftop, g)

def PopularTagsPerLang(df, lang, top_k = 10):
    """
    Function:
    Get top k tags with largest number of fanworks by media and in selected language.
    Input:
    - df: pandas.DataFrame.
    - lang: list[str], languages to include.
    - top_k: int, number of top tags to include.
    Output:
    - df_top: pandas.DataFrame.    
    """
    df_top = df.groupby(["MediaType","org_lang"])['fantom','cnt','MediaType',"org_lang"].apply(lambda x: x.nlargest(top_k, ['cnt'])).reset_index(drop=True)
    df_top['rank'] = df_top.sort_values(by=['cnt'], ascending=False).groupby(["MediaType","org_lang"]).cumcount()+1
    df_top = df_top.loc[df_top['org_lang'].isin(lang)].set_index(['rank','MediaType'])['fantom'].unstack(fill_value = 0)
    return(df_top)


def CleanStopWords(row, stop_lang_dict):
    """
    Function:
    Clean stop words.
    """
    if row['org_lang'] in stop_lang_dict:
        lang_stopwords = set(stopwords.words(stop_lang_dict[row['org_lang']])) | set(stopwords.words(stop_lang_dict['en']))
    else:
        lang_stopwords = []
    
    text = row['eng_text']
    text_clean = " ".join([w for w in text.split(" ") if w not in lang_stopwords])
    
    return(text_clean.lower())
    
def PrepEngText(df, stop_lang_dict):
    """
    Function:
    (Roughly) tokenize and remove stop words for English translation of tag.
    """
    df['eng_work'] = df['eng_ftm'].apply(lambda x: x.split(" - ")[0])

    reparent = re.compile(r"\(.+\)")
    reEn = re.compile(r"\b[a-zA-Z'-]+\b")

    df['eng_text'] = df['eng_work'].apply(lambda x: reparent.sub("", x))
    df['eng_text'] = df['eng_text'].apply(lambda x: " ".join(reEn.findall(x)).lower())
    df['eng_text'] = df.apply(lambda x: CleanStopWords(x, stop_lang_dict), axis = 1)
    
    df = df.loc[df['eng_text'].str.len() > 0]
    return(df)

def Vis_WordCloud(df, feature_value, max_word_per_cloud = 15, ncol = 4):
    """
    Function:
    Input:
    - df: pandas.DataFrame.
    - feature_value: dict, feature: value to display (e.g. "org_lang": ['en','fr','ja']).
    - max_word_per_cloud: int, maximum number of words to display for each word cloud.
    - ncol: int, number of column of subplots.
    Ouput:
    - fig: plot of WordCloud.
    """
    feature = list(feature_value.keys())[0]
    vals = feature_value[feature]
    
    ncol = min(ncol, len(vals)-1)    
    nrow = len(vals) // ncol + (len(vals) % ncol > 0)
    fig, axes = plt.subplots(nrows = nrow, ncols = ncol, figsize=(ncol*5, nrow*4))

    for il,val in enumerate(vals):
        tmp = df.loc[df[feature] == val]
        text = " ".join(list(tmp['eng_text']))

        lang_wordcloud = WordCloud(max_words = max_word_per_cloud, background_color = "white").generate(text)
        
        rid, cid = il // ncol, il % ncol
        axes[rid][cid].imshow(lang_wordcloud, interpolation = "bilinear")
        axes[rid][cid].axis("off")
        axes[rid][cid].set_title(val, fontsize = 14, fontweight = "bold")
    
    for j in range(il+1, nrow*ncol):
        fig.delaxes(axes[j // ncol][j % ncol])
    
    return(fig)
