

harmony_preprocess <- function(expr_mat_seurat, metadata, 
                              filter_genes = T, filter_cells = T,
                              normData = T, Datascaling = T, regressUMI = F, 
                              min_cells = 10, min_genes = 300, norm_method = "LogNormalize", scale_factor = 10000, 
                              b_x_low_cutoff = 0.0125, b_x_high_cutoff = 3, b_y_cutoff = 0.5, 
                              numVG = 300, npcs = 20, 
                              batch_label = "batchlb", celltype_label = "CellType")
{

  ##########################################################
  # preprocessing

  if(filter_genes == F) {
    min_cells = 0
  }
  if(filter_cells == F) {
    min_genes = 0
  }

  b_seurat <- CreateSeuratObject(expr_mat_seurat, meta.data = metadata, project = "Harmony_benchmark", 
                                 min.cells = min_cells, min.features = min_genes)
  if (normData) {
    b_seurat <- NormalizeData(object = b_seurat, normalization.method = norm_method, scale.factor = scale_factor)
  } else {
    b_seurat@data = b_seurat@raw.data
  }

#  b_seurat <- FindVariableFeatures(b_seurat, do.plot = TRUE, mean.function = ExpMean, dispersion.function = LogVMR, 
#                                x.low.cutoff = b_x_low_cutoff, x.high.cutoff = b_x_high_cutoff, y.cutoff = b_y_cutoff)
   b_seurat <- FindVariableFeatures(b_seurat,selection.method = "vst", nfeatures = 2000,verbose = FALSE)

#  print(paste("Num var genes: ", length(x = b_seurat@var.genes)))
#  if(length(x = b_seurat@var.genes) < numVG) {
#    print("The number of highly variable genes in data is small, can change the threshold to gain more HVG genes")
#  }

  if(regressUMI && Datascaling) {
    b_seurat <- ScaleData(object = b_seurat, vars.to.regress = c("nUMI"))  # in case of read count data
  } else if (Datascaling) { # default option
    b_seurat <- ScaleData(object = b_seurat)
  }

  return(b_seurat)
}

call_harmony <- function(b_seurat, batch_label, celltype_label, npcs = 20, 
                         plotout_dir = "", saveout_dir = "", 
                         outfilename_prefix = "", 
                         visualize = T, save_obj = T)
{

  #Harmony settings
  theta_harmony = 2
  numclust = 50
  max_iter_cluster = 100

  #plotting
  k_seed = 10
  tsne_perplex = 30
  tsneplot_filename = "_harmony_tsne"
  umapplot_filename = "_harmony_umap"
  obj_filename = "_harmony_sobj"
  pca_filename = "_harmony_pca"

  ##########################################################
  #run

  t1 = Sys.time()

  b_seurat <- RunPCA(object = b_seurat, pcs.compute = npcs, pc.genes = b_seurat@var.genes, do.print = TRUE, pcs.print = 1:5, 
                      genes.print = 5)

  b_seurat <- RunHarmony(object = b_seurat, batch_label, theta = theta_harmony, plot_convergence = TRUE, 
                          nclust = numclust, max.iter.cluster = max_iter_cluster)

  t2 = Sys.time()
  print(t2-t1)

  ##########################################################
  #save

  #harmony_res <- as.data.frame(b_seurat@dr$harmony@cell.embeddings)
  #cells_use <- rownames(harmony_res)

  #harmony_res$batchlb <- b_seurat@meta.data[, batch_label]
#  harmony_res$batch <- b_seurat@meta.data[, ]
 # harmony_res$celltype <- b_seurat@meta.data[, celltype_label]
 # write.table(harmony_res, file=paste0(saveout_dir,outfilename_prefix,pca_filename,".txt"), quote=F, sep='\t', row.names = T, col.names = NA)

  if(save_obj) {
    saveRDS(b_seurat, file=paste0(saveout_dir,outfilename_prefix,obj_filename,".RDS"))
  }

  if (visualize) {

    ##########################################################
    #preparing plots

    #b_seurat <- RunTSNE(b_seurat, reduction = "harmony", dims.use = 1:npcs, k.seed = 10, do.fast = T, check_duplicates = FALSE, perplexity = tsne_perplex)
    #b_seurat <- RunUMAP(b_seurat, reduction = "harmony", dims.use = 1:npcs, k.seed = 10, do.fast = T)
    b_seurat <- RunTSNE(b_seurat, reduction = "harmony", dims.use = 1:npcs, do.fast = T, k.seed = 10, check_duplicates = FALSE, perplexity = tsne_perplex)
    #b_seurat <- RunUMAP(b_seurat, reduction = "pca", dims.use = 1:npcs, do.fast = T, k.seed =10)

    ##########################################################
    #tSNE plot
    
    #p11 <- TSNEPlot(b_seurat, do.return = T, pt.size = 0.5, group.by = batch_label)
    #p12 <- TSNEPlot(b_seurat, do.return = T, pt.size = 0.5, group.by = celltype_label)

    p11 <- DimPlot(b_seurat, reduction = 'tsne', group.by = batch_label)
    p12 <- DimPlot(b_seurat, reduction = 'tsne', group.by = celltype_label)


    png(paste0(plotout_dir,outfilename_prefix,tsneplot_filename,".png"),width = 2*1000, height = 800, res = 2*72)
    print(plot_grid(p11, p12))
    dev.off()
  
    pdf(paste0(plotout_dir,outfilename_prefix,tsneplot_filename,".pdf"),width=15,height=7,paper='special') 
    print(plot_grid(p11, p12))
    dev.off()

    ##########################################################
    #UMAP plot

 #   p21 <- DimPlot(object = b_seurat, reduction = 'umap', group.by = batch_label)
 #   p22 <- DimPlot(object = b_seurat, reduction = 'umap', group.by = celltype_label)   # or orig.ident
#
#    png(paste0(plotout_dir,outfilename_prefix,umapplot_filename,".png"),width = 2*1000, height = 800, res = 2*72)
#    print(plot_grid(p21, p22))
#    dev.off()
#
#    pdf(paste0(plotout_dir,outfilename_prefix,umapplot_filename,".pdf"),width=15,height=7,paper='special') 
#    print(plot_grid(p21, p22))
#    dev.off()

  }

  return(b_seurat)
}







