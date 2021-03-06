D = /mss1/projects/AncReconTest/bin.new3/code/makeBlocks
F = config.file
P = _
T = /mss1/projects/AncReconTest/new3/simi/tree.txt

#all: Building.Blocks Orthology.Blocks Conserved.Segments 
pair: Grab.Data Building.Blocks Orthology.Blocks.pair Conserved.Segments.pair Genomes.Order

all: Grab.Data Building.Blocks Orthology.Blocks Conserved.Segments Genomes.Order


.PHONY: Grab.Data

# STEP 1
Grab.Data:
	@echo "=========== grabbing data from pairwise nets ==========="
	$D/readNets $F
	$D/getSegments $F

# STEP 2
Building.Blocks: $(wildcard *.processed.segs)
	@echo "======= partitioning genomes into building blocks ======"
	$D/partitionGenomes $F > $@

# STEP 3
Orthology.Blocks:
	@echo "=============== making orthology blocks ================"
	$D/makeOrthologyBlocks $F Building.Blocks > $Porthology.blocks.tmp
	awk '{if (NF > 2) {print $$1,$$2} else {print $$0}}' $Porthology.blocks.tmp > $@

Orthology.Blocks.pair:
	@echo "=============== making orthology blocks pair ================"
	$D/makeOrthologyBlocks.pair $F Building.Blocks > $Porthology.blocks.tmp
	awk '{if (NF > 2) {print $$1,$$2} else {print $$0}}' $Porthology.blocks.tmp > Orthology.Blocks 

# STEP 4
Conserved.Segments: Orthology.Blocks
	@echo "=== merging orthology blocks into conserved segments ==="
	$D/orthoBlocksToOrders $F $Porthology.blocks.tmp > $Porder.DS
	$D/makeConservedSegments $F $Porthology.blocks.tmp $Porder.DS > $Pconserved.segments.tmp
	$D/outgroupSegsToOrders $F $Pconserved.segments.tmp > $Porder.OG
	$D/cleanOutgroupSegs $F $Pconserved.segments.tmp $Porder.OG > $Pconserved.segments.tmp2
	awk '{if (NF > 2) {print $$1,$$2} else {print $$0}}' $Pconserved.segments.tmp2 > $PConserved.Segments 
	$D/makeTargetCS.pl $F $T $PConserved.Segments > $@ 

Conserved.Segments.pair: Orthology.Blocks
	@echo "=== merging orthology blocks into conserved segments pair ==="
	$D/orthoBlocksToOrders $F $Porthology.blocks.tmp > $Porder.DS
	$D/makeConservedSegments $F $Porthology.blocks.tmp $Porder.DS > $Pconserved.segments.tmp
	$D/outgroupSegsToOrders $F $Pconserved.segments.tmp > $Porder.OG
	$D/cleanOutgroupSegs $F $Pconserved.segments.tmp $Porder.OG > $Pconserved.segments.tmp2
	awk '{if (NF > 2) {print $$1,$$2} else {print $$0}}' $Pconserved.segments.tmp2 > Conserved.Segments 

# STEP 5
Genomes.Order: Conserved.Segments
	@echo "======== creating input files for inferring CARs ========"
	$D/createGenomeFile $F Conserved.Segments > $@

# remove intermediate files
tidy:
	rm -f $P*

clean:
	rm -rf Building.Blocks Conserved.Segments Orthology.Blocks Genomes.Order

