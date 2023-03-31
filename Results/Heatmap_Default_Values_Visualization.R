# set your working director using setwd()
setwd("your working directory")


library(reshape)
library(ggplot2)
library(gridExtra)
library(patchwork)


df_accuracy <- read.csv('Results_DefaultValues_Accuracy_Final.csv')
df_perception <- read.csv('Results_DefaultValue_Perception_Final.csv')


p_accuracy <- ggplot(df_accuracy, aes(x = DR, y = TM, fill = value)) +
  geom_tile(width = 1.0, position = 'identity') + 
  geom_text(aes(label = sprintf("%.2f", value)), color = "black", size = 3) + 
  scale_fill_gradientn(colors = rev(hcl.colors(7, "PuBu")), limits = c(0,0.25)) +
  guides(fill = guide_colourbar(barwidth = 0.5,
                                barheight = 20,
                                title = '',
                                reverse = TRUE)) + 
  theme(axis.ticks.length = unit(5, "pt"), 
        axis.text.x = element_text(size = 8, angle = 310, hjust = 0, colour = "black"),
        axis.text.y = element_text(size = 8, colour = "black"),
        panel.grid.major = element_blank(), 
        panel.background = element_blank(),
        legend.position = "none",
        legend.text = element_text(size=4),
        plot.margin = unit(c(0, 0, 0, 0), "pt")) + 
  labs(x = "", y = "") + 
  coord_equal()

p_accuracy

p_perception <- ggplot(df_perception, aes(x = DR, y = TM, fill = value)) +
  geom_tile(width = 1.0, position = 'identity') + 
  geom_text(aes(label = sprintf("%.2f", value)), color = "black", size = 3) + 
  scale_fill_gradientn(colors = rev(hcl.colors(7, "PuBu")), limits = c(0,0.25))+
  guides(fill = guide_colourbar(barwidth = 0.5,
                                barheight = 20,
                                title = '',
                                reverse = FALSE)) + 
  theme(axis.ticks.length = unit(5, "pt"),
        axis.ticks.y = element_blank(),
        axis.text.x = element_text(size = 8, angle = 310, hjust = 0, colour = "black"),
        axis.text.y = element_text(size = 0, colour = "black"),
        panel.grid.major = element_blank(), panel.background = element_blank(),
        legend.text=element_text(size=4),
        plot.margin = unit(c(0, 0, 0, 0), "pt")) + 
  labs(x = "", y = "") +
  coord_equal()
p_perception

p_accuracy + p_perception
