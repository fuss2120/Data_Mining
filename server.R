#
# This is the server logic of a Shiny web application. You can run the
# application by clicking 'Run App' above.
#
# Find out more about building applications with Shiny here:
#
#    http://shiny.rstudio.com/
#

library(shiny)
library(igraph)

# Define server logic required to draw a histogram
shinyServer(function(input, output) {
    output$distPlot <- renderPlot({
        
        if(input$Name){
            wiki = read.csv(file("/Users/michaelfuss/Documents/DMP_date.csv"), header=FALSE)
            relations <- data.frame(from=wiki$V1,
                                    to=wiki$V2,
                                    weights=wiki$V3,
                                    dates=wiki$V4)
            testSet <- subset(relations, relations$to==input$bins, select = c("from", "to", "weights", "dates"))
            g <- graph.data.frame(testSet, directed=FALSE)
            l <- layout.fruchterman.reingold(g, niter=5000, area=vcount(g)^4*10)
        }
        if(input$Date){
            wiki = read.csv(file("/Users/michaelfuss/Documents/DMP_date.csv"), header=FALSE)
            relations <- data.frame(from=wiki$V1,
                                    to=wiki$V2,
                                    weights=wiki$V3,
                                    dates=wiki$V4)
            testSet <- subset(relations, relations$dates>input$StartDate & relations$dates<input$EndDate , select = c("from", "to", "weights", "dates"))
            g <- graph.data.frame(testSet, directed=FALSE)
            l <- layout.fruchterman.reingold(g, niter=5000, area=vcount(g)^4*10)
        }

        # generate bins based on input$bins from ui.R

        # draw the histogram with the specified number of bins
        plot(g, layout = l, vertex.size=input$vSize,
             vertex.label.dist=0.8, vertex.label = V(g)$name, mark.groups = g$V3, vertex.label.color= "black", vertex.color=input$vfcolor, edge.label.color=input$efcolor,edge.arrow.size=input$eLWD)
    
    })

})
