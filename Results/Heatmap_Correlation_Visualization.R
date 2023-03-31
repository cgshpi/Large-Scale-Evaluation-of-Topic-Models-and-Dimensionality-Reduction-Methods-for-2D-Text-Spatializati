# set your working director using setwd()
# setwd("your working directory")



# Important note: In the file Results_Correlation_Final.csv we replaced
# the entries 'Silhouette coefficient' by 'Silhouette Coefficient'
# and the entries 'Distance consistency' by 'Distance Consistency'
# using an usual Text Editor


library(reshape)
library(ggplot2)
library(gridExtra)
library(patchwork)
library(RColorBrewer)

df <- read.csv('Results_Correlation_Final.csv')

colors <- brewer.pal(9, "BrBG")

ggplot(df, aes(x = Variable.1, y = Variable.2, fill = Correlation)) +
  geom_tile(width = 1.0, position = 'identity') + 
  geom_text(aes(label = sprintf("%.2f", Correlation)), color = "black", size = 4) + 
  scale_fill_gradient2(low = colors[1], mid = colors[5], high = colors[9],
                       midpoint = 0.0, na.value = "grey", limits = c(-1,1)) +
  guides(fill = guide_colourbar(barwidth = 1,
                                barheight = 10,
                                title = '',
                                #label = FALSE,
                                #ticks = FALSE,
                                reverse = FALSE)) +
  theme(axis.ticks.length = unit(5, "pt"),
        axis.text.y = element_text(size = 10),
        axis.text.x = element_text(size = 10, angle = 330, hjust = 0, colour = "black"),
        panel.grid.major = element_blank(), panel.background = element_blank(),
        legend.text=element_text(size=8)) +
  scale_x_discrete(limits=c('Trustworthiness', 'Continuity', 'Shephard Diagram Correlation', '7-Neighborhood Hit',
                            'Davies-Bouldin-Index Normalized', 'Calinski-Harabasz-Index Normalized', 'Distance Consistency',
                            'Silhouette Coefficient')) +
  scale_y_discrete(limits=c('Silhouette Coefficient', 'Distance Consistency', 'Calinski-Harabasz-Index Normalized',
                            'Davies-Bouldin-Index Normalized', '7-Neighborhood Hit','Shephard Diagram Correlation',
                            'Continuity','Trustworthiness'))+
  labs(x = "", y = "") +
  coord_equal()

  