# set your working director using setwd()
setwd("your working directory")


library(ggplot2)

df_accuracy <- read.csv("Results_Boxplot_Accuracy_Final.csv")
df_perception <- read.csv("Results_Boxplot_Perception_Final.csv")

ggplot(df_accuracy,aes(x = TM,ymin = min,lower = q1,middle = median,
              upper = q3,ymax = max,
           fill = TM,
           group = TM)) +
  geom_boxplot(stat = "identity",fill ='#3690c0' ) +
  scale_x_discrete(limits=c("(VSM,-,X)", "(VSM,+,X)",
                            "(NMF,-,-)","(NMF,-,+)", "(NMF,+,-)", "(NMF,+,+)",
                            "(LSI,-,-)", "(LSI,-,+)", "(LSI,+,-)",
                            "(LSI,+,+)", "(LDA,X,-)", "(LDA,X,+)",
                            "(BERT,X,X)")) + 
  labs(x="", y="") + 
  theme(axis.text.x = element_text(size = 8, angle = 330, hjust = 0, colour = "black")) + 
  facet_grid(DR~.,scales="free_x")
