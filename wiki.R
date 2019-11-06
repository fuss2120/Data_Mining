library (igraph)

wiki<-read.csv(file.choose(),header=FALSE)
g<-graph.data.frame(wiki, directed=FALSE)
bipartite.mapping(g)
plot(g, vertex.label.cex = .8, vertex.label.color = "black")

##Warning message:
##In length(vattrs[[name]]) <- vc : length of NULL cannot be changed
V(g)$color <- ifelse(V(g)$type, "red", "blue")

##Warning message:
##In length(vattrs[[name]]) <- vc : length of NULL cannot be changed
V(g)$shape <- ifelse(V(g)$type, "circle", "square")

E(g)$color <- "lightgray"

plot(g, vertex.label.cex = .8, vertex.label.color = "black")

V(g)$label.color <- "black" 
V(g)$label.cex <- 1
V(g)$frame.color <-  "gray"
V(g)$size <- 18

plot(g, layout = layout_with_graphopt)

##Error in v(graph) : Not a bipartite graph, supply `types' argument
plot(g, layout=layout.bipartite, vertex.size=7, vertex.label.cex=0.6)

types <- V(g)$type 
deg <- degree(g)
bet <- betweenness(g)
clos <- closeness(g)
eig <- eigen_centrality(g)$vector

## Error in data.frame(types, deg, bet, clos, eig) : arguments imply differing number of rows: 0, 11381
cent_df <- data.frame(types, deg, bet, clos, eig)

##Error: object 'cent_df' not found
cent_df[order(cent_df$type, decreasing = TRUE),]
