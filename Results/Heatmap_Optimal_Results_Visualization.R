# set your working director using setwd()
setwd("your working directory")


library(reshape)
library(ggplot2)
library(gridExtra)
library(patchwork)

df_accuracy <- read.csv('Results_Heatmap_Accuracy_Final.csv')
df_perception <- read.csv('Results_Heatmap_Perception_Final.csv')



p_accuracy <- ggplot(df_accuracy, aes(x = Dataset, y = Layout, fill = value)) +
  geom_tile(width = 1.0, position = 'identity') + 
  geom_text(aes(label = value), color = "black", size = 1) + 
  scale_fill_gradientn(colors = rev(hcl.colors(7, "PuBu")), limits = c(0,1)) +
  guides(fill = guide_colourbar(barwidth = 0.5,
                                barheight = 20,
                                title = '',
                                reverse = TRUE)) + 
  theme(axis.ticks.length = unit(5, "pt"), 
        axis.text.x = element_text(size = 4, angle = 330, hjust = 0, colour = "black"),
        axis.text.y = element_text(size = 4, colour = "black"),
        panel.grid.major = element_blank(), 
        panel.background = element_blank(),
        legend.position = "none",
        legend.text = element_text(size=4),
        plot.margin = unit(c(0, 0, 0, 0), "pt")) + 
  labs(x = "", y = "") + # if you need to change the labels for axis
  scale_y_discrete(limits=c('(BERT,X,UMAP,X)',
                            '(BERT,X,TSNE,X)',
                            '(BERT,X,SOM,X)',
                            '(BERT,X,MDS,X)',
                            '(LDA,X,UMAP,+)',
                            '(LDA,X,TSNE,+)',
                            '(LDA,X,SOM,+)',
                            '(LDA,X,MDS,+)',
                            '(LDA,X,UMAP,-)',
                            '(LDA,X,TSNE,-)',
                            '(LDA,X,SOM,-)',
                            '(LDA,X,MDS,-)',
                            '(NMF,+,UMAP,+)',
                            '(NMF,+,TSNE,+)',
                            '(NMF,+,SOM,+)',
                            '(NMF,+,MDS,+)',
                            '(NMF,-,UMAP,+)',
                            '(NMF,-,TSNE,+)',
                            '(NMF,-,SOM,+)',
                            '(NMF,-,MDS,+)',
                            '(NMF,+,UMAP,-)',
                            '(NMF,+,TSNE,-)',
                            '(NMF,+,SOM,-)',
                            '(NMF,+,MDS,-)',
                            '(NMF,-,UMAP,-)',
                            '(NMF,-,TSNE,-)',
                            '(NMF,-,SOM,-)',
                            '(NMF,-,MDS,-)',
                            '(LSI,+,UMAP,+)',
                            '(LSI,+,TSNE,+)',
                            '(LSI,+,SOM,+)',
                            '(LSI,+,MDS,+)',
                            '(LSI,-,UMAP,+)',
                            '(LSI,-,TSNE,+)',
                            '(LSI,-,SOM,+)',
                            '(LSI,-,MDS,+)',
                            '(LSI,+,UMAP,-)',
                            '(LSI,+,TSNE,-)',
                            '(LSI,+,SOM,-)',
                            '(LSI,+,MDS,-)',
                            '(LSI,-,UMAP,-)',
                            '(LSI,-,TSNE,-)',
                            '(LSI,-,SOM,-)',
                            '(LSI,-,MDS,-)',
                            '(VSM,+,UMAP,X)',
                            '(VSM,+,TSNE,X)',
                            '(VSM,+,SOM,X)',
                            '(VSM,+,MDS,X)',
                            '(VSM,-,UMAP,X)', 
                            '(VSM,-,TSNE,X)',
                            '(VSM,-,SOM,X)',
                            '(VSM,-,MDS,X)'))+
  coord_equal()

p_accuracy

p_perception <- ggplot(df_perception, aes(x = Dataset, y = Layout, fill = value)) +
  geom_tile(width = 1.0, position = 'identity') + 
  geom_text(aes(label = value), color = "black", size = 1) + 
  scale_fill_gradientn(colors = rev(hcl.colors(7, "PuBu")), limits = c(0,1))+
  guides(fill = guide_colourbar(barwidth = 0.5,
                                barheight = 20,
                                title = '',
                                #label = FALSE,
                                #ticks = FALSE,
                                reverse = FALSE)) + 
  theme(axis.ticks.length = unit(5, "pt"),
        axis.ticks.y = element_blank(),
        axis.text.x = element_text(size = 4, angle = 330, hjust = 0, colour = "black"),
        axis.text.y = element_text(size = 0, colour = "black"),
        panel.grid.major = element_blank(), panel.background = element_blank(),
        legend.text=element_text(size=4),
        plot.margin = unit(c(0, 0, 0, 0), "pt")) + 
  labs(x = "", y = "") + 
  scale_y_discrete(limits=c('(BERT,X,UMAP,X)',
                            '(BERT,X,TSNE,X)',
                            '(BERT,X,SOM,X)',
                            '(BERT,X,MDS,X)',
                            '(LDA,X,UMAP,+)',
                            '(LDA,X,TSNE,+)',
                            '(LDA,X,SOM,+)',
                            '(LDA,X,MDS,+)',
                            '(LDA,X,UMAP,-)',
                            '(LDA,X,TSNE,-)',
                            '(LDA,X,SOM,-)',
                            '(LDA,X,MDS,-)',
                            '(NMF,+,UMAP,+)',
                            '(NMF,+,TSNE,+)',
                            '(NMF,+,SOM,+)',
                            '(NMF,+,MDS,+)',
                            '(NMF,-,UMAP,+)',
                            '(NMF,-,TSNE,+)',
                            '(NMF,-,SOM,+)',
                            '(NMF,-,MDS,+)',
                            '(NMF,+,UMAP,-)',
                            '(NMF,+,TSNE,-)',
                            '(NMF,+,SOM,-)',
                            '(NMF,+,MDS,-)',
                            '(NMF,-,UMAP,-)',
                            '(NMF,-,TSNE,-)',
                            '(NMF,-,SOM,-)',
                            '(NMF,-,MDS,-)',
                            '(LSI,+,UMAP,+)',
                            '(LSI,+,TSNE,+)',
                            '(LSI,+,SOM,+)',
                            '(LSI,+,MDS,+)',
                            '(LSI,-,UMAP,+)',
                            '(LSI,-,TSNE,+)',
                            '(LSI,-,SOM,+)',
                            '(LSI,-,MDS,+)',
                            '(LSI,+,UMAP,-)',
                            '(LSI,+,TSNE,-)',
                            '(LSI,+,SOM,-)',
                            '(LSI,+,MDS,-)',
                            '(LSI,-,UMAP,-)',
                            '(LSI,-,TSNE,-)',
                            '(LSI,-,SOM,-)',
                            '(LSI,-,MDS,-)',
                            '(VSM,+,UMAP,X)',
                            '(VSM,+,TSNE,X)',
                            '(VSM,+,SOM,X)',
                            '(VSM,+,MDS,X)',
                            '(VSM,-,UMAP,X)', 
                            '(VSM,-,TSNE,X)',
                            '(VSM,-,SOM,X)',
                            '(VSM,-,MDS,X)'))+
  coord_equal()
p_perception

p_accuracy + p_perception
