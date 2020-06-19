#!/usr/bin/perl -w
# This script convert kaldi's nnet3 to nnet1. The supported structure including:
# lstm, tdnn, fsmn, dfsmn.
# Other nnet layers will be added later.
# Date: Thur. Dec 11 2018 -- wangzhichao214232@sogou-inc.com
#########################################################################################
if(@ARGV!=2)
{
  print "Usage::perl convert_nnet3_to_nnet1.pl <model-in> <model-out> \n";
  exit 1;
}

################### Set net config ######################
#$NumComponents=4;          #for lstm set-up
#$NumComponents=31;         #for tdnn-lstm set-up
#$NumComponents=24;         #for FSMN online set-up
#$NumComponents=60;         #for FSMN offline set-up
#$NumComponents=15;         #for 3TDNN-3BLSTM set-up
$NumComponents=31;         #for 9TDNN-4BLSTM set-up
#$NumComponents=46;          #for 12TDNN-10Attention set-up

################## 3TDNN-3BLSTM example ##################
#system("cat <<EOF > nnet.proto
#name=tdnn1 type=NaturalGradientAffineComponent input=355 output=1024
#name=relu1 type=RectifiedLinearComponent 
#name=renorm1 type=NormalizeComponent
#name=splice1 type=Splice offset=[-1,0,1] input=1024 output=3072
#name=tdnn2 type=NaturalGradientAffineComponent input=3072 output=1024
#name=relu2 type=RectifiedLinearComponent
#name=renorm2 type=NormalizeComponent
#name=splice2 type=Splice offset=[-1,0,1] input=1024 output=3072
#name=tdnn3 type=NaturalGradientAffineComponent input=3072 output=1024
#name=relu3 type=RectifiedLinearComponent
#name=renorm3 type=NormalizeComponent
#name=blstm1 type=Blstm input=1024 cell=1536 output=768 discard=0
#name=blstm2 type=Blstm input=768 cell=1536 output=768 discard=0
#name=blstm3 type=Blstm input=768 cell=1536 output=768 discard=0
#name=output type=NaturalGradientAffineComponent input=768 output=3766
#EOF");
################## 9TDNN-4BLSTM example ##################
system("cat <<EOF > nnet.proto
name=tdnn1 type=NaturalGradientAffineComponent input=355 output=1280
name=relu1 type=RectifiedLinearComponent 
name=renorm1 type=NormalizeComponent
name=splice1 type=Splice offset=[-1,0,1] input=1280 output=3840
name=tdnn2 type=NaturalGradientAffineComponent input=3840 output=1280
name=relu2 type=RectifiedLinearComponent
name=renorm2 type=NormalizeComponent
name=splice2 type=Splice offset=[-1,0,1] input=1280 output=3840
name=tdnn3 type=NaturalGradientAffineComponent input=3840 output=1024
name=relu3 type=RectifiedLinearComponent
name=renorm3 type=NormalizeComponent
name=blstm1 type=Blstm input=1024 cell=1280 output=768 discard=0
name=tdnn4 type=NaturalGradientAffineComponent input=768 output=1536
name=relu4 type=RectifiedLinearComponent
name=renorm4 type=NormalizeComponent
name=splice3 type=Splice offset=[-1,0,1] input=1536 output=4608
name=tdnn5 type=LinearComponent input=4608 output=512
name=blstm2 type=Blstm input=512 cell=1280 output=768 discard=0
name=tdnn6 type=NaturalGradientAffineComponent input=768 output=1536
name=relu6 type=RectifiedLinearComponent
name=renorm6 type=NormalizeComponent
name=splice4 type=Splice offset=[-1,0,1] input=1536 output=4608
name=tdnn7 type=LinearComponent input=4608 output=512
name=blstm3 type=Blstm input=512 cell=1280 output=768 discard=0
name=tdnn8 type=NaturalGradientAffineComponent input=768 output=1536
name=relu8 type=RectifiedLinearComponent
name=renorm8 type=NormalizeComponent
name=splice5 type=Splice offset=[-1,0,1] input=1536 output=4608
name=tdnn9 type=LinearComponent input=4608 output=512
name=blstm4 type=Blstm input=512 cell=1280 output=768 discard=0
name=output type=NaturalGradientAffineComponent input=768 output=4528
EOF");
################### Deep FSMN offline example ###################
#system("cat <<EOF > nnet.proto
#name=affine1 type=NaturalGradientAffineComponent input=355 output=1536
#name=relu1 type=RectifiedLinearComponent 
#name=batchnorm1 type=BatchNormComponent
#name=linear1 type=LinearComponent input=1536 output=512
#name=fsmn1 type=Fsmn input=512 output=512 l-order=3 r-order=1 stride=1 depend=0
#name=affine2 type=NaturalGradientAffineComponent input=512 output=1536
#name=relu2 type=RectifiedLinearComponent 
#name=batchnorm2 type=BatchNormComponent
#name=linear2 type=LinearComponent input=1536 output=512
#name=fsmn2 type=Fsmn input=512 output=512 l-order=3 r-order=0 stride=1 depend=0
#name=affine3 type=NaturalGradientAffineComponent input=512 output=1536
#name=relu3 type=RectifiedLinearComponent 
#name=batchnorm3 type=BatchNormComponent
#name=linear3 type=LinearComponent input=1536 output=512
#name=fsmn3 type=Fsmn input=512 output=512 l-order=3 r-order=1 stride=1 depend=-2
#name=affine4 type=NaturalGradientAffineComponent input=512 output=1536
#name=relu4 type=RectifiedLinearComponent 
#name=batchnorm4 type=BatchNormComponent
#name=linear4 type=LinearComponent input=1536 output=512
#name=fsmn4 type=Fsmn input=512 output=512 l-order=2 r-order=0 stride=1 depend=-2
#name=affine5 type=NaturalGradientAffineComponent input=512 output=1536
#name=relu5 type=RectifiedLinearComponent 
#name=batchnorm5 type=BatchNormComponent
#name=linear5 type=LinearComponent input=1536 output=512
#name=fsmn5 type=Fsmn input=512 output=512 l-order=1 r-order=1 stride=1 depend=-2
#name=affine6 type=NaturalGradientAffineComponent input=512 output=1536
#name=relu6 type=RectifiedLinearComponent 
#name=batchnorm6 type=BatchNormComponent
#name=linear6 type=LinearComponent input=1536 output=512
#name=fsmn6 type=Fsmn input=512 output=512 l-order=2 r-order=0 stride=1 depend=-2
#name=affine7 type=NaturalGradientAffineComponent input=512 output=1536
#name=relu7 type=RectifiedLinearComponent 
#name=batchnorm7 type=BatchNormComponent
#name=linear7 type=LinearComponent input=1536 output=512
#name=fsmn7 type=Fsmn input=512 output=512 l-order=1 r-order=1 stride=1 depend=-2
#name=affine8 type=NaturalGradientAffineComponent input=512 output=1536
#name=relu8 type=RectifiedLinearComponent 
#name=batchnorm8 type=BatchNormComponent
#name=linear8 type=LinearComponent input=1536 output=512
#name=fsmn8 type=Fsmn input=512 output=512 l-order=2 r-order=0 stride=1 depend=-2
#name=affine9 type=NaturalGradientAffineComponent input=512 output=1536
#name=relu9 type=RectifiedLinearComponent 
#name=batchnorm9 type=BatchNormComponent
#name=linear9 type=LinearComponent input=1536 output=512
#name=fsmn9 type=Fsmn input=512 output=512 l-order=2 r-order=0 stride=1 depend=-2
#name=affine10 type=NaturalGradientAffineComponent input=512 output=1536
#name=relu10 type=RectifiedLinearComponent 
#name=batchnorm10 type=BatchNormComponent
#name=linear10 type=LinearComponent input=1536 output=512
#name=fsmn10 type=Fsmn input=512 output=512 l-order=2 r-order=0 stride=1 depend=-2
#name=affine11 type=NaturalGradientAffineComponent input=512 output=1536
#name=relu11 type=RectifiedLinearComponent 
#name=batchnorm11 type=BatchNormComponent
#name=linear11 type=LinearComponent input=1536 output=512
#name=prefinal.affine type=NaturalGradientAffineComponent input=512 output=1536
#name=prefinal.relu type=RectifiedLinearComponent
#name=prefinal.batchnorm1 type=BatchNormComponent
#name=prefinal.linear type=LinearComponent input=1536 output=256
#name=prefinal.batchnorm2 type=BatchNormComponent
#name=output type=NaturalGradientAffineComponent input=256 output=3766
#EOF");

################### Deep FSMN online example ###################
#system("cat <<EOF > nnet.proto
#name=affine1 type=NaturalGradientAffineComponent input=355 output=1536
#name=relu1 type=RectifiedLinearComponent 
#name=batchnorm1 type=BatchNormComponent
#name=linear1 type=LinearComponent input=1536 output=512
#name=fsmn1 type=Fsmn input=512 output=512 l-order=3 r-order=1 stride=1
#name=dfsmn2 type=Dfsmn input=512 hid=1536 output=512 l-order=3 r-order=0 stride=1
#name=dfsmn3 type=Dfsmn input=512 hid=1536 output=512 l-order=3 r-order=1 stride=1
#name=dfsmn4 type=Dfsmn input=512 hid=1536 output=512 l-order=2 r-order=0 stride=1
#name=dfsmn5 type=Dfsmn input=512 hid=1536 output=512 l-order=1 r-order=1 stride=1
#name=dfsmn6 type=Dfsmn input=512 hid=1536 output=512 l-order=2 r-order=0 stride=1
#name=dfsmn7 type=Dfsmn input=512 hid=1536 output=512 l-order=1 r-order=1 stride=1
#name=dfsmn8 type=Dfsmn input=512 hid=1536 output=512 l-order=2 r-order=0 stride=1
#name=dfsmn9 type=Dfsmn input=512 hid=1536 output=512 l-order=2 r-order=0 stride=1
#name=dfsmn10 type=Dfsmn input=512 hid=1536 output=512 l-order=2 r-order=0 stride=1
#name=affine2 type=NaturalGradientAffineComponent input=512 output=1536
#name=relu2 type=RectifiedLinearComponent 
#name=batchnorm2 type=BatchNormComponent
#name=linear2 type=LinearComponent input=1536 output=512
#name=prefinal.affine type=NaturalGradientAffineComponent input=512 output=1536
#name=prefinal.relu type=RectifiedLinearComponent
#name=prefinal.batchnorm1 type=BatchNormComponent
#name=prefinal.linear type=LinearComponent input=1536 output=256
#name=prefinal.batchnorm2 type=BatchNormComponent
#name=output type=NaturalGradientAffineComponent input=256 output=3766
#EOF");

################### 12TDNN-10ATTENTION example ###################
#system("cat <<EOF > nnet.proto
#name=batchnorm0 type=BatchNormComponent blocks=5
#name=tdnn1 type=NaturalGradientAffineComponent input=355 output=1280
#name=relu1 type=RectifiedLinearComponent
#name=batchnorm1 type=BatchNormComponent
#name=splice1 type=Splice offset=[-1,0,1] input=1280 output=3840
#name=tdnn2 type=NaturalGradientAffineComponent input=3840 output=1280
#name=relu2 type=RectifiedLinearComponent
#name=batchnorm2 type=BatchNormComponent
#name=splice2 type=Splice offset=[-1,0,1] input=1280 output=3840
#name=tdnn3 type=NaturalGradientAffineComponent input=3840 output=1024
#name=relu3 type=RectifiedLinearComponent
#name=batchnorm3 type=BatchNormComponent
#name=positionembedding1 type=PositionEmbeddingComponent
#name=attentionblock1 type=MultiHeadSelfAttentionBlock input=1024 hid=3072 output=1024 stride=1 scale=1.0
#name=feedforwardblock1 type=FeedForwardBlock input=1024 hid=2048 proj=512 offset=[-1,0,1] output=1024 scale=1.0
#name=positionembedding2 type=PositionEmbeddingComponent
#name=attentionblock2 type=MultiHeadSelfAttentionBlock input=1024 hid=3072 output=1024 stride=1 scale=1.0
#name=feedforwardblock2 type=FeedForwardBlock input=1024 hid=2048 proj=512 offset=[-1,0,1] output=1024 scale=1.0
#name=positionembedding3 type=PositionEmbeddingComponent
#name=attentionblock3 type=MultiHeadSelfAttentionBlock input=1024 hid=3072 output=1024 stride=1 scale=1.0
#name=feedforwardblock3 type=FeedForwardBlock input=1024 hid=2048 proj=512 offset=[-1,0,1] output=1024 scale=1.0
#name=positionembedding4 type=PositionEmbeddingComponent
#name=attentionblock4 type=MultiHeadSelfAttentionBlock input=1024 hid=3072 output=1024 stride=1 scale=1.0
#name=feedforwardblock4 type=FeedForwardBlock input=1024 hid=2048 proj=512 offset=[-1,0,1] output=1024 scale=1.0
#name=positionembedding5 type=PositionEmbeddingComponent
#name=attentionblock5 type=MultiHeadSelfAttentionBlock input=1024 hid=3072 output=1024 stride=1 scale=1.0
#name=feedforwardblock5 type=FeedForwardBlock input=1024 hid=2048 proj=512 offset=[-1,0,1] output=1024 scale=1.0
#name=positionembedding6 type=PositionEmbeddingComponent
#name=attentionblock6 type=MultiHeadSelfAttentionBlock input=1024 hid=3072 output=1024 stride=1 scale=1.0
#name=feedforwardblock6 type=FeedForwardBlock input=1024 hid=2048 proj=512 offset=[-1,0,1] output=1024 scale=1.0
#name=positionembedding7 type=PositionEmbeddingComponent
#name=attentionblock7 type=MultiHeadSelfAttentionBlock input=1024 hid=3072 output=1024 stride=1 scale=1.0
#name=feedforwardblock7 type=FeedForwardBlock input=1024 hid=2048 proj=512 offset=[-1,0,1] output=1024 scale=1.0
#name=positionembedding8 type=PositionEmbeddingComponent
#name=attentionblock8 type=MultiHeadSelfAttentionBlock input=1024 hid=3072 output=1024 stride=1 scale=1.0
#name=feedforwardblock8 type=FeedForwardBlock input=1024 hid=2048 proj=512 offset=[-1,0,1] output=1024 scale=1.0
#name=positionembedding9 type=PositionEmbeddingComponent
#name=attentionblock9 type=MultiHeadSelfAttentionBlock input=1024 hid=3072 output=1024 stride=1 scale=1.0
#name=feedforwardblock9 type=FeedForwardBlock input=1024 hid=2048 proj=512 offset=[-1,0,1] output=1024 scale=1.0
#name=positionembedding10 type=PositionEmbeddingComponent
#name=attentionblock10 type=MultiHeadSelfAttentionBlock input=1024 hid=3072 output=1024 stride=1 scale=1.0
#name=feedforwardblock10 type=FeedForwardBlock input=1024 hid=2048 proj=512 offset=[-1,0,1] output=1024 scale=1.0
#name=prefinal type=NaturalGradientAffineComponent input=1024 output=1024
#name=relu4 type=RectifiedLinearComponent
#name=batchnorm4 type=BatchNormComponent
#name=output type=NaturalGradientAffineComponent input=1024 output=4528
#EOF");

################### tdnn-lstm example ###################
#system("cat <<EOF > nnet.proto
#name=tdnn1 type=NaturalGradientAffineComponent input=355 output=1024
#name=relu1 type=RectifiedLinearComponent 
#name=renorm1 type=NormalizeComponent
#name=splice1 type=Splice offset=[-1,0,1] input=1024 output=3072
#name=tdnn2 type=NaturalGradientAffineComponent input=3072 output=1024
#name=relu2 type=RectifiedLinearComponent
#name=renorm2 type=NormalizeComponent
#name=splice2 type=Splice offset=[-1,0,1] input=1024 output=3072
#name=tdnn3 type=NaturalGradientAffineComponent input=3072 output=1024
#name=relu3 type=RectifiedLinearComponent
#name=renorm3 type=NormalizeComponent
#name=lstm1 type=Lstm input=1024 cell=2048 output=768 discard=4
#name=splice3 type=Splice offset=[-1,0,1] input=768 output=2304
#name=tdnn4 type=NaturalGradientAffineComponent input=2304 output=1024
#name=relu4 type=RectifiedLinearComponent
#name=renorm4 type=NormalizeComponent
#name=splice4 type=Splice offset=[-1,0,1] input=1024 output=3072
#name=tdnn5 type=NaturalGradientAffineComponent input=3072 output=1024
#name=relu5 type=RectifiedLinearComponent
#name=renorm5 type=NormalizeComponent
#name=lstm2 type=Lstm input=1024 cell=2048 output=512 discard=8
#name=splice5 type=Splice offset=[-1,0,1] input=512 output=1536
#name=tdnn6 type=NaturalGradientAffineComponent input=1536 output=1024
#name=relu6 type=RectifiedLinearComponent
#name=renorm6 type=NormalizeComponent
#name=splice6 type=Splice offset=[-1,0,1] input=1024 output=3072
#name=tdnn7 type=NaturalGradientAffineComponent input=3072 output=1024
#name=relu7 type=RectifiedLinearComponent
#name=renorm7 type=NormalizeComponent
#name=lstm3 type=Lstm input=1024 cell=2048 output=512 discard=12
#name=output type=NaturalGradientAffineComponent input=512 output=3766
#EOF");

################### lstm example #######################
#system("cat <<EOF > nnet.proto
#name=lstm1 type=Lstm input=355 cell=2560 output=768 discard=0
#name=lstm2 type=Lstm input=768 cell=2560 output=768 discard=0
#name=lstm3 type=Lstm input=768 cell=2560 output=768 discard=0
#name=output type=NaturalGradientAffineComponent input=768 output=3766
#EOF");
################### Net conf end ########################
$model_in=$ARGV[0];
$model_out=$ARGV[1];

open PROTO, "<nnet.proto" or die "$!";
open IN, "<$model_in" or die "$!";
open OUT, ">$model_out" or die "$!";

print OUT "<Nnet>\n";

$layer_cnt = 0;
while($layer_cnt < $NumComponents)
{
  $component = <PROTO>;
  chomp $component;
  print "Layer:$layer_cnt\n";
  if($component=~/AffineComponent/) {
    &parse_affine($component);
  }
  elsif($component=~/=LinearComponent/) { ### "=" is to fix the conflict with RectifiedLinearComponent
    &parse_linear($component);
  }
  elsif($component=~/RectifiedLinearComponent/) {
    &parse_relu($component);
  }  
  elsif($component=~/PositionEmbeddingComponent/) {
    &parse_pe($component);
  }
  elsif($component=~/NormalizeComponen/) { 
    &parse_renorm($component); 
  } 
  elsif($component=~/BatchNormComponent/) { ### "=" is to fix the conflict with BlockBatchNormComponent
    &parse_batchnorm($component);
  }
  elsif($component=~/Splice/) {
    &parse_splice($component);
  }
  elsif($component=~/Fsmn/) {
    &parse_fsmn($component);
  }
  elsif($component=~/Dfsmn/) {
    &parse_dfsmn($component); 
  }
  elsif($component=~/MultiHeadSelfAttentionBlock/) {
    &parse_mhablock($component); 
  }
  elsif($component=~/FeedForwardBlock/) {
    &parse_feedforwardblock($component); 
  }
  elsif($component=~/Lstm/) {
    &parse_lstm($component);
  }
  elsif($component=~/Blstm/) {
    &parse_blstm($component);
  }else {
    print "Error: $layer_cnt+1 th Component no support - $component\n";
    exit 1;
  }
  $layer_cnt++;
} 
print OUT "</Nnet>\n"; print "Success! Converting finished!\n";

sub parse_node {
  @a = split /=/, $_[0];
  return $a[1];
}

sub parse_affine {
  $find = 0;
  @units = split /\s+/, $_[0];
  $input_dim = &parse_node($units[2]);
  $output_dim = &parse_node($units[3]);

  @linear_affine=();   # $output_dim * $input_dim;
  
  while($line=<IN>)
  {  
    if($line=~/ComponentName/ && $line=~/AffineComponent/)
    {
      $find = 1;
      $cnt=0;
      while($cnt < $output_dim)
      {
        $line=<IN>;
        chomp $line;
        @params=split /\s+/, $line;
        push @linear_affine, @params[1..$input_dim];
        $cnt++;
      }

      # write to nnet1 model
      print OUT "<AffineTransform> $output_dim $input_dim\n";
      print OUT "<LearnRateCoef> 2.5 <BiasLearnRateCoef> 2.5 <MaxNorm> 0\n";
      print OUT " [\n";
      $cnt=0;
      while($cnt < $output_dim)
      {
        if ($cnt == ($output_dim-1))
        {
          print OUT "  @linear_affine[$cnt*$input_dim..($cnt+1)*$input_dim-1] ]\n";
          last;
        }
        print OUT "  @linear_affine[$cnt*$input_dim..($cnt+1)*$input_dim-1]\n"; 
        $cnt++;
      }

      $line=<IN>;
      chomp $line;
      @params = split /\s+/, $line;
      shift @params;
      @affine_bias=();   # 1 * $output_dim; 
      push @affine_bias, @params[1..$output_dim];
      print OUT " [ @affine_bias ]\n";
      last;
    }
  }
  if($find == 0)
  {
    print "Error: Can't find $_[0] in nnet3 model file.\n";
    exit 1;
  }
  print "converting  $_[0] finished...\n";
}

sub parse_linear {
  $find = 0;
  @units = split /\s+/, $_[0];
  $input_dim = &parse_node($units[2]);
  $output_dim = &parse_node($units[3]);

  @linear_affine=();   # $output_dim * $input_dim;
  
  while($line=<IN>)
  {  
    if($line=~/ComponentName/ && $line=~/<LinearComponent>/)
    {
      $find = 1;
      $cnt=0;
      while($cnt < $output_dim)
      {
        $line=<IN>;
        chomp $line;
        @params=split /\s+/, $line;
        push @linear_affine, @params[1..$input_dim];
        $cnt++;
      }

      # write to nnet1 model
      print OUT "<LinearTransform> $output_dim $input_dim\n";
      print OUT "<LearnRateCoef> 2.5 <MaxNorm> 0\n";
      print OUT " [\n";
      $cnt=0;
      while($cnt < $output_dim)
      {
        if ($cnt == ($output_dim-1))
        {
          print OUT "  @linear_affine[$cnt*$input_dim..($cnt+1)*$input_dim-1] ]\n";
          last;
        }
        print OUT "  @linear_affine[$cnt*$input_dim..($cnt+1)*$input_dim-1]\n"; 
        $cnt++;
      }
      last;
    }
  }
  if($find == 0)
  {
    print "Error: Can't find $_[0] in nnet3 model file.\n";
    exit 1;
  }
  print "converting  $_[0] finished...\n";
}

sub parse_relu {
  $find = 0;
  
  while($line=<IN>)
  {
    chomp $line;
    if($line=~/ComponentName/ && $line=~/RectifiedLinearComponent/)
    {
      $find = 1;
      @a = split /\s+/, $line;
      $dim = $a[4];
      print OUT "<ReLU> $dim $dim\n";
      last;
    }
  }
  if($find == 0)
  {
    print "Error: Can't find $_[0] in nnet3 model file.\n";
    exit 1;
  }
  print "converting  $_[0] finished...\n";
}

sub parse_pe {
  $find = 0;
  while($line=<IN>) 
  {
    chomp $line;
    if($line=~/ComponentName/ && $line=~/PositionEmbeddingComponent/) 
    {
      $find = 1;
      @a = split /\s+/, $line;
      $input_dim = $a[4];
      $output_dim = $a[6];
      print OUT "<PositionEmbeddingComponent> $output_dim $input_dim\n";
      last;
    }
  }
  if($find == 0)
  {
    print "Error: Can't find $_[0] in nnet3 model file.\n";
    exit 1;
  }
  print "converting $_[0] finished...\n";
}

sub parse_renorm {
  $find=0;
  while($line=<IN>)
  {
    chomp $line;
    if($line=~/ComponentName/ && $line=~/NormalizeComponent/)
    {
      $find = 1;
      @a = split /\s+/, $line;
      $dim = $a[4];
      $target_rms = $a[6];
      print OUT "<NormalizeComponent> $dim $dim\n";
      print OUT "<TargetRms> $target_rms\n";
      last;
    }
  }
  if($find == 0)
  {
    print "Error: Can't find $_[0] in nnet3 model file.\n";
    exit 1;
  }
  print "converting  $_[0] finished...\n";
}

sub parse_batchnorm {
  $find=0;
  $num_blocks = 1;
  @units = split /\s+/, $_[0];
  if(@units == 3)
  {
    $num_blocks = &parse_node($units[2]);
  }
  @stat_mean = ();
  @stat_var = ();
  while($line=<IN>)
  {
    chomp $line;
    if($line=~/ComponentName/ && $line=~/BatchNormComponent/)
    {
      $find = 1;
      @a = split /\s+/, $line;
      $dim = $a[4];
      $block_dim = $a[6];
      if($dim == $block_dim && $num_blocks != 1)
      {
        $dim = $dim * $num_blocks;                
      }
      $epsilon = $a[8];
      $target_rms = $a[10];
      $count = $a[14];
      $cnt = 0;
      while($cnt < 16) {
	shift @a;
	$cnt++;
      }
      push @stat_mean, @a[1..$block_dim];
      $line = <IN>;
      chomp $line;
      @params = split /\s+/, $line;
      shift @params;
      push @stat_var, @params[1..$block_dim];
	  
      ### write nnet1 batchnorm ###
      print OUT "<BatchNormComponent> $dim $dim\n";
      print OUT "<BlockDim> $block_dim <Epsilon> $epsilon <TargetRms> $target_rms <Count> $count\n";
      print OUT "[ @stat_mean ]\n";
      print OUT "[ @stat_var ]\n";
      last;
    }
  }
  if($find == 0)
  {
    print "Error: Can't find $_[0] in nnet3 model file.\n";
    exit 1;
  }
  print "converting  $_[0] finished...\n";
}

sub parse_splice {
  @units = split /\s+/, $_[0];
  $offset = &parse_node($units[2]);
  $input_dim = &parse_node($units[3]);
  $output_dim = &parse_node($units[4]);
  print OUT "<Splice> $output_dim $input_dim\n";
  $offset =~ s/\[(.*)\]/$1/;
  @splice_idx = split /,/, $offset;
  print OUT "[ @splice_idx ]\n";
  
  print "converting  $_[0] finished...\n";
}

sub parse_fsmn {
  $find = 0;
  @units = split /\s+/, $_[0];
  $input_dim = &parse_node($units[2]);
  $output_dim = &parse_node($units[3]);
  $l_order = &parse_node($units[4]);
  $r_order = &parse_node($units[5]);
  $stride = &parse_node($units[6]);
  $depend = &parse_node($units[7]);
  @filter_params = ();
  while($line=<IN>) 
  {
    chomp $line;
    if($line=~/ComponentName/ && $line=~/NaturalGradientPerElementScaleComponent/)	
	{
	  $find = 1;
	  @a = split /\s+/, $line;
	  $line=<IN>;
	  
	  #read the SumBlockComponent and check the filter size
	  $line=<IN>;          
	  chomp $line;
	  @b = split /\s+/, $line;
	  $sum_block_in = $b[4];
	  if ($sum_block_in != ($l_order+$r_order+1)*$input_dim)
	  {
	    print "Error: the filter order in $_[0] may be wrong!\n";
		exit 1;
	  }
	  $cnt = 0;
	  while($cnt < 10)
	  {
	    shift @a;
		$cnt++;
	  }
	  push @filter_params, @a[1..$input_dim*($l_order+$r_order+1)];
	  
	  ####### write nnet1 fsmn ########	  
	  print OUT "<Fsmn> $output_dim $input_dim\n";
	  print OUT "<LearnRateCoef> 2.5 <LOrder> $l_order <ROrder> $r_order <Stride> $stride <Depend> $depend\n";
      print OUT " [\n";
      $cnt=0;
      while($cnt < $l_order+$r_order+1)
      {
        if ($cnt == ($l_order+$r_order))
        {
          print OUT "  @filter_params[$cnt*$input_dim..($cnt+1)*$input_dim-1] ]\n";
          last;
        }
        print OUT "  @filter_params[$cnt*$input_dim..($cnt+1)*$input_dim-1]\n"; 
        $cnt++;
      }
	  last;
	}
  }
  if($find == 0)
  {
    print "Error: Can't find $_[0] in nnet3 model file.\n";
    exit 1;
  }
  print "converting  $_[0] finished...\n";
}

sub parse_dfsmn {
  $find = 0;
  @units = split /\s+/, $_[0];
  $input_dim = &parse_node($units[2]);
  $hid_dim = &parse_node($units[3]);
  $output_dim = &parse_node($units[4]);
  $l_order = &parse_node($units[5]);
  $r_order = &parse_node($units[6]);
  $stride = &parse_node($units[7]);
  $epsilon = 0;
  $target_rms = 0;
  $count = 0;
  @linear_affine = ();
  @affine_bias = ();
  @stat_mean = ();
  @stat_var = ();
  @linear_project = ();
  @filter_params = ();
  while($line=<IN>) 
  {
    chomp $line;
    if($line=~/ComponentName/ && $line=~/NaturalGradientAffineComponent/) {
      ### reading the affine-transform part ###
      $cnt=0;
      while($cnt < $hid_dim)
      {
        $line=<IN>;
        chomp $line;
        @params=split /\s+/, $line;
        push @linear_affine, @params[1..$input_dim];
        $cnt++;
      }
	  
      $line=<IN>;
      chomp $line;
      @params = split /\s+/, $line;
      shift @params;
      @affine_bias=();   # 1 * $hid_dim; 
      push @affine_bias, @params[1..$hid_dim];
	  
      ### reading the relu part ###
      $line=<IN>;
      $line=<IN>;
      chomp $line;
      if(!($line=~/ComponentName/ && $line=~/RectifiedLinearComponent/))
      {
	print "Error: RELU part in DFSMN is not found!\n";
        $find = 0;
	last;
      }
      
      ### reading the batchnorm part ###
      $line=<IN>; $line=<IN>; $line=<IN>;
      $line=<IN>;
      chomp $line;
      if(!($line=~/ComponentName/ && $line=~/BatchNormComponent/))
      {
	print "Error: BatchNorm part in DFSMN is not found!\n";
	$find = 0;
	last;  
      }
      @a = split /\s+/, $line;
      $epsilon = $a[8];
      $target_rms = $a[10];
      $count = $a[14];
      $cnt = 0;
      while($cnt < 16) {
        shift @a;
        $cnt++;
      }
      push @stat_mean, @a[1..$hid_dim];
      $line = <IN>;
      chomp $line;
      @params = split /\s+/, $line;
      shift @params;
      push @stat_var, @params[1..$hid_dim];
	  
      ### reading the linear-project part ###
      $line=<IN>;
      $line=<IN>;
      chomp $line;
      if(!($line=~/ComponentName/ && $line=~/LinearComponent/))
      {
	print "Error: Linear-project part in DFSMN is not found!\n";
	$find = 0;
	last;
      }
      $cnt=0;
      while($cnt < $output_dim)
      {
        $line=<IN>;
        chomp $line;
        @params=split /\s+/, $line;
        push @linear_project, @params[1..$hid_dim];
        $cnt++;
      }	
	  
      ### reading the fsmn part ###
      $line=<IN>;
      $line=<IN>;
      chomp $line;
      if(!($line=~/ComponentName/ && $line=~/NaturalGradientPerElementScaleComponent/))	
      {
        print "Error: the FSMN part in DFSMN is not found!\n";
	$find = 0;
	last;
      }
      @a = split /\s+/, $line;
      $line=<IN>;	  
      #read the SumBlockComponent and check the filter size
      $line=<IN>;          
      chomp $line;
      @b = split /\s+/, $line;
      $sum_block_in = $b[4];
      if ($sum_block_in != ($l_order+$r_order+1)*$input_dim)
      {
	print "Error: the filter order in $_[0] may be wrong!\n";
	exit 1;
      }
      $cnt = 0;
      while($cnt < 10)
      {
	shift @a;
	$cnt++;
      }
      push @filter_params, @a[1..$input_dim*($l_order+$r_order+1)];
	  
      ### write nne1 dfsmn ###
      $find = 1;
      print OUT "<DeepFsmn> $output_dim $input_dim\n";
      print OUT "<LearnRateCoef> 2.5 <HidSize> $hid_dim <LOrder> $l_order <ROrder> $r_order <Stride> $stride <Epsilon> $epsilon <TargetRms> $target_rms <Count> $count\n";
	  
      ### write affine-transform part ###
      print OUT " [\n";
      $cnt=0;
      while($cnt < $hid_dim)
      {
        if ($cnt == ($hid_dim-1))
        {
          print OUT "  @linear_affine[$cnt*$input_dim..($cnt+1)*$input_dim-1] ]\n";
          last;
        }
        print OUT "  @linear_affine[$cnt*$input_dim..($cnt+1)*$input_dim-1]\n"; 
        $cnt++;
      }	  
      print OUT " [ @affine_bias ]\n";
	  
      ### write RELU part ###
	  
      ### write batchnorm part ###
      print OUT "[ @stat_mean ]\n";
      print OUT "[ @stat_var ]\n";
	  
      ### write linear-project part ###
      $cnt=0;
      while($cnt < $output_dim)
      {
        if ($cnt == ($output_dim-1))
        {
          print OUT "  @linear_project[$cnt*$hid_dim..($cnt+1)*$hid_dim-1] ]\n";
          last;
        }
        print OUT "  @linear_project[$cnt*$hid_dim..($cnt+1)*$hid_dim-1]\n"; 
        $cnt++;
      }	  
	  
      ### write fsmn part ###
      $cnt=0;
      while($cnt < $l_order+$r_order+1)
      {
        if ($cnt == ($l_order+$r_order))
        {
          print OUT "  @filter_params[$cnt*$input_dim..($cnt+1)*$input_dim-1] ]\n";
          last;
        }
        print OUT "  @filter_params[$cnt*$input_dim..($cnt+1)*$input_dim-1]\n"; 
        $cnt++;
      }
      last;
    }
  }
  if($find == 0)
  {
    print "Error: Can't find $_[0] in nnet3 model file.\n";
    exit 1;
  }
  print "converting  $_[0] finished...\n";
}

sub parse_mhablock {
  $find=0;
  @units = split /\s+/, $_[0];
  $input_dim = &parse_node($units[2]);
  $hid_dim = &parse_node($units[3]);
  $output_dim = &parse_node($units[4]);
  $time_stride = &parse_node($units[5]);
  $sum_scale = &parse_node($units[6]);
  $num_heads = 0;
  $key_dim = 0;
  $value_dim = 0;
  $key_scale = 0;
  $num_leftinput = 0;
  $num_rightinput = 0;
  $epsilon = 0;
  $target_rms = 0;
  $count = 0;
  @linear_affine = ();
  @affine_bias = ();
  @stat_mean = ();
  @stat_var = ();
  
  while($line=<IN>)  {
    chomp $line;
    if($line=~/ComponentName/ && $line=~/NaturalGradientAffineComponent/) {
      ### reading the affine-transform part ###
      $cnt=0;
      while($cnt < $hid_dim)
      {
        $line=<IN>;
        chomp $line;
        @params=split /\s+/, $line;
        push @linear_affine, @params[1..$input_dim];
        $cnt++;
      }

      $line=<IN>;
      chomp $line;
      @params = split /\s+/, $line;
      shift @params;
      @affine_bias=();   # 1 * $hid_dim; 
      push @affine_bias, @params[1..$hid_dim];
     
      ### reading the self-attention part ###
      $line = <IN>;
      $line = <IN>; 
      chomp $line;
      if(!($line=~/ComponentName/ && $line=~/Contextless3RestrictedAttentionComponent/))
      {
        print "Error: SelfAttention part in MultiHeadAttentionBlock is not found!\n";
        $find = 0;
        last;
      }
      @a = split /\s+/, $line;
      $num_heads = $a[4];
      $key_dim = $a[6];
      $value_dim = $a[8];
      $num_leftinput = $a[10];
      $num_rightinput = $a[12];
      $key_scale = $a[26];
      
      ### reading the batchnorm part ###
      $line=<IN>;
      chomp $line;
      if(!($line=~/ComponentName/ && $line=~/BatchNormComponent/))
      {
        print "Error: BatchNorm part in MultiHeadAttentionBlock is not found!\n";
        $find = 0;
        last;
      }
      @a = split /\s+/, $line;
      $epsilon = $a[8];
      $target_rms = $a[10];
      $count = $a[14];
      $cnt = 0;
      while($cnt < 16) {
        shift @a;
        $cnt++;
      }
      push @stat_mean, @a[1..$output_dim];
      $line = <IN>;
      chomp $line;
      @params = split /\s+/, $line;
      shift @params;
      push @stat_var, @params[1..$output_dim];

      ### write nnet1 MultiHeadAttentionBlock model ###
      $find = 1;
      print OUT "<MultiHeadSelfAttentionBlock> $output_dim $input_dim\n";
      print OUT "<NumHeads> $num_heads <KeyDim> $key_dim <ValueDim> $value_dim <QueryDim> $key_dim <KeyScale> $key_scale <TimeStride> $time_stride \
<NumLeftInuts> $num_leftinput <NumRightInputs> $num_rightinput <Epsilon> $epsilon <TargetRms> $target_rms <Count> $count <SumScale> $sum_scale\n";

      ### write affine-transform part ###
      print OUT " [\n";
      $cnt=0;
      while($cnt < $hid_dim)
      {
        if ($cnt == ($hid_dim-1))
        {
          print OUT "  @linear_affine[$cnt*$input_dim..($cnt+1)*$input_dim-1] ]\n";
          last;
        }
        print OUT "  @linear_affine[$cnt*$input_dim..($cnt+1)*$input_dim-1]\n";
        $cnt++;
      }
      print OUT " [ @affine_bias ]\n";
      
     ### write batchnorm part ###
      print OUT "[ @stat_mean ]\n";
      print OUT "[ @stat_var ]\n";
      last;
    }#end if
  }#end while

  if($find == 0)
  {
    print "Error: Can't find $_[0] in nnet3 model file.\n";
    exit 1;
  }
  print "converting $_[0] finished...\n";
}

sub parse_feedforwardblock {
  $find=0;
  @units = split /\s+/, $_[0];
  $input_dim = &parse_node($units[2]);
  $hid_dim = &parse_node($units[3]);
  $project_dim = &parse_node($units[4]);
  $offset = &parse_node($units[5]);
  $output_dim = &parse_node($units[6]);
  $sum_scale = &parse_node($units[7]);
  $offset =~ s/\[(.*)\]/$1/;
  @splice_idx = split /,/, $offset;

  $epsilon = 0;
  $target_rms = 0;
  $count = 0;
  @linear_affine = ();
  @affine_bias = ();
  @linear_project1 = ();
  @linear_project2 = ();
  @stat_mean = ();
  @stat_var = ();

  while($line=<IN>)  {
    chomp $line;
    if($line=~/ComponentName/ && $line=~/NaturalGradientAffineComponent/) {
      ### reading affine-transform part ###
      $cnt=0;
      while($cnt < $hid_dim)
      {
        $line=<IN>;
        chomp $line;
        @params=split /\s+/, $line;
        push @linear_affine, @params[1..$input_dim];
        $cnt++;
      }

      $line=<IN>;
      chomp $line;
      @params = split /\s+/, $line;
      shift @params;
      @affine_bias=();   # 1 * $hid_dim; 
      push @affine_bias, @params[1..$hid_dim]; 
   
      ### reading the relu part ###
      $line=<IN>;
      $line=<IN>;
      chomp $line;
      if(!($line=~/ComponentName/ && $line=~/RectifiedLinearComponent/))
      {
        print "Error: RELU part in FeedForwardBlock is not found!\n";
        $find = 0;
        last;
      }

      ### reading the linear-project-1 part ###
      $line=<IN>; $line=<IN>; $line=<IN>;
      $line=<IN>;
      chomp $line;
      if(!($line=~/ComponentName/ && $line=~/LinearComponent/))
      {
        print "Error: Linear-project-1 part in FeedForwardBlock is not found!\n";
        $find = 0;
        last;
      }
      $cnt=0;
      while($cnt < $project_dim)
      {
        $line=<IN>;
        chomp $line;
        @params=split /\s+/, $line;
        push @linear_project1, @params[1..$hid_dim];
        $cnt++;
      }

      ### reading the linear-project-2 part ###
      $line=<IN>;
      $line=<IN>;
      chomp $line;
      if(!($line=~/ComponentName/ && $line=~/LinearComponent/))
      {
        print "Error: Linear-project-2 part in FeedForwardBlock is not found!\n";
        $find = 0;
        last;
      }
      $cnt=0;
      $splice_dim = @splice_idx;
      $tdnn_dim = $splice_dim * $project_dim;
      while($cnt < $output_dim)
      {
        $line=<IN>;
        chomp $line;
        @params=split /\s+/, $line;
   
        push @linear_project2, @params[1..$tdnn_dim];
        $cnt++;
      }
      ### reading the batchnorm part ###
      $line=<IN>;
      $line=<IN>;
      chomp $line;
      if(!($line=~/ComponentName/ && $line=~/BatchNormComponent/))
      {
        print "Error: BatchNorm part in MultiHeadAttentionBlock is not found!\n";
        $find = 0;
        last;
      }
      @a = split /\s+/, $line;
      $epsilon = $a[8];
      $target_rms = $a[10];
      $count = $a[14];
      $cnt = 0;
      while($cnt < 16) {
        shift @a;
        $cnt++;
      }
      push @stat_mean, @a[1..$output_dim];
      $line = <IN>;
      chomp $line;
      @params = split /\s+/, $line;
      shift @params;
      push @stat_var, @params[1..$output_dim];

      ### write nnet1 FeedForwardBlock model ###
      $find = 1;
      print OUT "<FeedForwardBlock> $output_dim $input_dim\n";
      print OUT "<HidSize> $hid_dim <ProjectDim> $project_dim <Splice> [ @splice_idx ] <Epsilon> $epsilon <TargetRms> $target_rms <Count> $count <SumScale> $sum_scale\n";

      ### write affine-transform part ###
      print OUT " [\n";
      $cnt=0;
      while($cnt < $hid_dim)
      {
        if ($cnt == ($hid_dim-1))
        {
          print OUT "  @linear_affine[$cnt*$input_dim..($cnt+1)*$input_dim-1] ]\n";
          last;
        }
        print OUT "  @linear_affine[$cnt*$input_dim..($cnt+1)*$input_dim-1]\n";
        $cnt++;
      }
      print OUT " [ @affine_bias ]\n";

      ### write linear-project-1 part ###
      print OUT " [\n";
      $cnt=0;
      while($cnt < $project_dim)
      {
        if ($cnt == ($project_dim-1))
        {
          print OUT "  @linear_project1[$cnt*$hid_dim..($cnt+1)*$hid_dim-1] ]\n";
          last;
        }
        print OUT "  @linear_project1[$cnt*$hid_dim..($cnt+1)*$hid_dim-1]\n";
        $cnt++;
      }

      ### write linear-project-2 part ###
      print OUT " [\n";
      $cnt=0;
      while($cnt < $output_dim)
      {
        if ($cnt == ($output_dim-1))
        {
          print OUT "  @linear_project2[$cnt*$tdnn_dim..($cnt+1)*$tdnn_dim-1] ]\n";
          last;
        }
        print OUT "  @linear_project2[$cnt*$tdnn_dim..($cnt+1)*$tdnn_dim-1]\n";
        $cnt++;
      }
  
      ### write batchnorm part ###
      print OUT "[ @stat_mean ]\n";
      print OUT "[ @stat_var ]\n";
      last;
    }#end if
  }#end while
  if($find == 0)
  {
    print "Error: Can't find $_[0] in nnet3 model file.\n";
    exit 1;
  }
  print "converting $_[0] finished...\n";
}

sub parse_lstm {
  $find=0;
  @units = split /\s+/, $_[0];
  $input_dim = &parse_node($units[2]);
  $cell_dim = &parse_node($units[3]);
  $output_dim = &parse_node($units[4]);
  $discard = &parse_node($units[5]); 

  while($line=<IN>)
  {
    chomp $line;
    if($line=~/ComponentName/ && $line=~/W_all/)
    {
      $find=1;	    
    # read ifco_x and ifco_r
      @ix=();   # $cell_dim * $input_dim
      @ir=();   # $cell_dim * $output_dim
      @fx=();
      @fr=();
      @cx=();
      @cr=();
      @ox=();
      @or=();
      $gate_cnt=0;
      while($gate_cnt < 4)
      {
        $cell_cnt = 0;
        while($cell_cnt < $cell_dim)
	{
	  $line=<IN>;
	  chomp $line;
	  @params = split /\s+/, $line;
	  if($gate_cnt == 0)
	  {
	     push @ix, @params[1..$input_dim];
	     push @ir, @params[$input_dim+1..$input_dim+$output_dim];
	  }
	  if($gate_cnt == 1)
	  {
             push @fx, @params[1..$input_dim];
             push @fr, @params[$input_dim+1..$input_dim+$output_dim];
	  }
          if($gate_cnt == 2)
          {
             push @cx, @params[1..$input_dim];
             push @cr, @params[$input_dim+1..$input_dim+$output_dim];
          }
          if($gate_cnt == 3)
          {
             push @ox, @params[1..$input_dim];
             push @or, @params[$input_dim+1..$input_dim+$output_dim];
          }
	  $cell_cnt++;
	}
	$gate_cnt++;
      }

      ########## write nne1 cifo_x ###########
      print OUT "<LstmProjectedNnet3Streams> $output_dim $input_dim\n";
      print OUT "<CellDim> $cell_dim <ClipGradient> 5 <DiscardInput> $discard\n";
      print OUT " [\n";
      $cnt = 0;
      while($cnt < $cell_dim)
      {
	print OUT "  @cx[$cnt*$input_dim..($cnt+1)*$input_dim-1]\n";
        $cnt++;
      }
      $cnt = 0;
      while($cnt < $cell_dim)
      {
	print OUT "  @ix[$cnt*$input_dim..($cnt+1)*$input_dim-1]\n";
	$cnt++;
      }
      $cnt = 0;
      while($cnt < $cell_dim)
      {
        print OUT "  @fx[$cnt*$input_dim..($cnt+1)*$input_dim-1]\n";
        $cnt++;
      }
      $cnt = 0;
      while($cnt < $cell_dim)
      {
	if ($cnt == ($cell_dim-1))
	{
	  print OUT "  @ox[$cnt*$input_dim..($cnt+1)*$input_dim-1] ]\n";
	  last;
	}
        print OUT "  @ox[$cnt*$input_dim..($cnt+1)*$input_dim-1]\n";
        $cnt++;
      }

      ######### write nnet1 cifo_r ##########
      print OUT " [\n";
      $cnt = 0;
      while($cnt < $cell_dim)
      {
        print OUT "  @cr[$cnt*$output_dim..($cnt+1)*$output_dim-1]\n";
        $cnt++;
      }
      $cnt = 0;
      while($cnt < $cell_dim)
      {
        print OUT "  @ir[$cnt*$output_dim..($cnt+1)*$output_dim-1]\n";
        $cnt++;
      }
      $cnt = 0;
      while($cnt < $cell_dim)
      {
        print OUT "  @fr[$cnt*$output_dim..($cnt+1)*$output_dim-1]\n";
        $cnt++;
      }
      $cnt = 0;
      while($cnt < $cell_dim)
      {
        if ($cnt == ($cell_dim-1))
        {
          print OUT "  @or[$cnt*$output_dim..($cnt+1)*$output_dim-1] ]\n";
          last;
        }
	print OUT "  @or[$cnt*$output_dim..($cnt+1)*$output_dim-1]\n";
        $cnt++;
      }

#      if ($layer_cnt==3)
#      { 
#	print "@ix[0..$input_dim-1]\n@ir[0..$output_dim-1]\n";
#	print "@ix[$input_dim*($cell_dim-1)..$input_dim*$cell_dim-1]\n@ir[$output_dim*($cell_dim-1)..$output_dim*$cell_dim-1]\n";
#        exit 1; 
#      }
         
      ######## read ifco_bias and write to nnet1 as cifo_bias #########
      $line=<IN>;
      chomp $line;
      @params = split /\s+/, $line;
      shift @params;
      @i_bias=();   # 1 * $cell_dim
      @f_bias=();
      @c_bias=();
      @o_bias=();
      push @i_bias, @params[1..$cell_dim];
      push @f_bias, @params[$cell_dim+1..$cell_dim*2];
      push @c_bias, @params[$cell_dim*2+1..$cell_dim*3];
      push @o_bias, @params[$cell_dim*3+1..$cell_dim*4];
      
      print OUT " [ @c_bias @i_bias @f_bias @o_bias ]\n";

  
#      if($layer_cnt == 3)
#      { 
#      print "@i_bias\n@f_bias\n\n@c_bias\n@o_bias\n";    
#      exit 1;
#      }

      ######### read ifo_c and write to nnet1 #######
      $line=<IN>;      #read useless line;
      $line=<IN>;      #read useless line;
    
      # read i_c;
      @i_c=();
      $line=<IN>;    
      chomp $line;
      @params = split /\s+/, $line;
      push @i_c, @params[1..$cell_dim];  
 
      # read f_c;
      @f_c=();
      $line=<IN>;
      chomp $line;
      @params = split /\s+/, $line;
      push @f_c, @params[1..$cell_dim]; 

      # read o_c;
      @o_c=();
      $line=<IN>;
      chomp $line;
      @params = split /\s+/, $line;
      push @o_c, @params[1..$cell_dim];

      print OUT " [ @i_c ]\n";
      print OUT " [ @f_c ]\n";
      print OUT " [ @o_c ]\n";
      
      # read until W_rp (r_m for nnet1)
      while($line=<IN>)
      {
        chomp $line;
        if ($line=~/ComponentName/ && $line=~/W_rp/)
        {
          last;
        }
      }
   
      ######## read W_rp and rp_bias and write to nnet1 #########
      $cnt=0;
      @r_m=();    # $output_dim * $cell_dim
      while($cnt < $output_dim)
      {
        $line=<IN>;
        chomp $line;
        @params = split /\s+/, $line;
        push @r_m, @params[1..$cell_dim];
        $cnt++;
      }
   
      print OUT " [\n";
      $cnt=0;
      while($cnt < $output_dim)
      {
        if ($cnt == ($output_dim-1))
        {
          print OUT "  @r_m[$cnt*$cell_dim..($cnt+1)*$cell_dim-1] ]\n";
          last;
        }
        print OUT "  @r_m[$cnt*$cell_dim..($cnt+1)*$cell_dim-1]\n";
        $cnt++;
      }      
#      if($layer_cnt==3)
#      {
#        print "@r_m[0..$cell_dim-1]\n@r_m[$cell_dim*($output_dim-1)..$cell_dim*$output_dim-1]\n";
#        exit 1;
#      }
      # read rm_bias
      $line=<IN>;
      chomp $line;
      @params = split /\s+/, $line;
      shift @params;
      @rm_bias=();
      push @rm_bias, @params[1..$output_dim]; 

      print OUT " [ @rm_bias ]\n"; 
#      print OUT "<!EndOfComponent>\n";
#      if($layer_cnt==3)
#      {
#        print "@rm_bias\n";
#	exit 1;
#      }
      last;
    }
  }
  if ($find == 0) {
    print "Error: Can't find $_[0] in nnet3 model file.\n";
    exit 1;
  }
  print "converting  $_[0] finished...\n";
}

sub parse_blstm {
  $find=0;
  @units = split /\s+/, $_[0];
  $input_dim = &parse_node($units[2]);
  $cell_dim = &parse_node($units[3]);
  $output_dim = &parse_node($units[4]);
  $output_dim = $output_dim / 2;
  $discard = &parse_node($units[5]); 

  while($line=<IN>)
  {
    chomp $line;
    if($line=~/ComponentName/ && $line=~/forward.W_all/)
    {
      $find=1;	    
    # read ifco_x and ifco_r
      @ix=();   # $cell_dim * $input_dim
      @ir=();   # $cell_dim * $output_dim
      @fx=();
      @fr=();
      @cx=();
      @cr=();
      @ox=();
      @or=();
      $gate_cnt=0;
      while($gate_cnt < 4)
      {
        $cell_cnt = 0;
        while($cell_cnt < $cell_dim)
	{
	  $line=<IN>;
	  chomp $line;
	  @params = split /\s+/, $line;
	  if($gate_cnt == 0)
	  {
	     push @ix, @params[1..$input_dim];
	     push @ir, @params[$input_dim+1..$input_dim+$output_dim];
	  }
	  if($gate_cnt == 1)
	  {
             push @fx, @params[1..$input_dim];
             push @fr, @params[$input_dim+1..$input_dim+$output_dim];
	  }
          if($gate_cnt == 2)
          {
             push @cx, @params[1..$input_dim];
             push @cr, @params[$input_dim+1..$input_dim+$output_dim];
          }
          if($gate_cnt == 3)
          {
             push @ox, @params[1..$input_dim];
             push @or, @params[$input_dim+1..$input_dim+$output_dim];
          }
	  $cell_cnt++;
	}
	$gate_cnt++;
      }

      ########## write nne1 cifo_x ###########
      $bidirection_output_dim = $output_dim * 2;
      print OUT "<BlstmProjectedNnet3Streams> $bidirection_output_dim $input_dim\n";
#      print OUT "<CellDim> $cell_dim <ClipGradient> 5 <DiscardInput> $discard\n";
      print OUT "<CellDim> $cell_dim <ClipGradient> 5\n";
      print OUT " [\n";
      $cnt = 0;
      while($cnt < $cell_dim)
      {
	print OUT "  @cx[$cnt*$input_dim..($cnt+1)*$input_dim-1]\n";
        $cnt++;
      }
      $cnt = 0;
      while($cnt < $cell_dim)
      {
	print OUT "  @ix[$cnt*$input_dim..($cnt+1)*$input_dim-1]\n";
	$cnt++;
      }
      $cnt = 0;
      while($cnt < $cell_dim)
      {
        print OUT "  @fx[$cnt*$input_dim..($cnt+1)*$input_dim-1]\n";
        $cnt++;
      }
      $cnt = 0;
      while($cnt < $cell_dim)
      {
	if ($cnt == ($cell_dim-1))
	{
	  print OUT "  @ox[$cnt*$input_dim..($cnt+1)*$input_dim-1] ]\n";
	  last;
	}
        print OUT "  @ox[$cnt*$input_dim..($cnt+1)*$input_dim-1]\n";
        $cnt++;
      }

      ######### write nnet1 cifo_r ##########
      print OUT " [\n";
      $cnt = 0;
      while($cnt < $cell_dim)
      {
        print OUT "  @cr[$cnt*$output_dim..($cnt+1)*$output_dim-1]\n";
        $cnt++;
      }
      $cnt = 0;
      while($cnt < $cell_dim)
      {
        print OUT "  @ir[$cnt*$output_dim..($cnt+1)*$output_dim-1]\n";
        $cnt++;
      }
      $cnt = 0;
      while($cnt < $cell_dim)
      {
        print OUT "  @fr[$cnt*$output_dim..($cnt+1)*$output_dim-1]\n";
        $cnt++;
      }
      $cnt = 0;
      while($cnt < $cell_dim)
      {
        if ($cnt == ($cell_dim-1))
        {
          print OUT "  @or[$cnt*$output_dim..($cnt+1)*$output_dim-1] ]\n";
          last;
        }
	print OUT "  @or[$cnt*$output_dim..($cnt+1)*$output_dim-1]\n";
        $cnt++;
      }

#      if ($layer_cnt==3)
#      { 
#	print "@ix[0..$input_dim-1]\n@ir[0..$output_dim-1]\n";
#	print "@ix[$input_dim*($cell_dim-1)..$input_dim*$cell_dim-1]\n@ir[$output_dim*($cell_dim-1)..$output_dim*$cell_dim-1]\n";
#        exit 1; 
#      }
         
      ######## read ifco_bias and write to nnet1 as cifo_bias #########
      $line=<IN>;
      chomp $line;
      @params = split /\s+/, $line;
      shift @params;
      @i_bias=();   # 1 * $cell_dim
      @f_bias=();
      @c_bias=();
      @o_bias=();
      push @i_bias, @params[1..$cell_dim];
      push @f_bias, @params[$cell_dim+1..$cell_dim*2];
      push @c_bias, @params[$cell_dim*2+1..$cell_dim*3];
      push @o_bias, @params[$cell_dim*3+1..$cell_dim*4];
      
      print OUT " [ @c_bias @i_bias @f_bias @o_bias ]\n";

  
#      if($layer_cnt == 3)
#      { 
#      print "@i_bias\n@f_bias\n\n@c_bias\n@o_bias\n";    
#      exit 1;
#      }

      ######### read ifo_c and write to nnet1 #######
      $line=<IN>;      #read useless line;
      $line=<IN>;      #read useless line;
    
      # read i_c;
      @i_c=();
      $line=<IN>;    
      chomp $line;
      @params = split /\s+/, $line;
      push @i_c, @params[1..$cell_dim];  
 
      # read f_c;
      @f_c=();
      $line=<IN>;
      chomp $line;
      @params = split /\s+/, $line;
      push @f_c, @params[1..$cell_dim]; 

      # read o_c;
      @o_c=();
      $line=<IN>;
      chomp $line;
      @params = split /\s+/, $line;
      push @o_c, @params[1..$cell_dim];

      print OUT " [ @i_c ]\n";
      print OUT " [ @f_c ]\n";
      print OUT " [ @o_c ]\n";
      
      # read until W_rp (r_m for nnet1)
      while($line=<IN>)
      {
        chomp $line;
        if ($line=~/ComponentName/ && $line=~/W_rp/)
        {
          last;
        }
      }
   
      ######## read W_rp and rp_bias and write to nnet1 #########
      $cnt=0;
      @r_m=();    # $output_dim * $cell_dim
      while($cnt < $output_dim)
      {
        $line=<IN>;
        chomp $line;
        @params = split /\s+/, $line;
        push @r_m, @params[1..$cell_dim];
        $cnt++;
      }
   
      print OUT " [\n";
      $cnt=0;
      while($cnt < $output_dim)
      {
        if ($cnt == ($output_dim-1))
        {
          print OUT "  @r_m[$cnt*$cell_dim..($cnt+1)*$cell_dim-1] ]\n";
          last;
        }
        print OUT "  @r_m[$cnt*$cell_dim..($cnt+1)*$cell_dim-1]\n";
        $cnt++;
      }      
#      if($layer_cnt==3)
#      {
#        print "@r_m[0..$cell_dim-1]\n@r_m[$cell_dim*($output_dim-1)..$cell_dim*$output_dim-1]\n";
#        exit 1;
#      }
      # read rm_bias
      $line=<IN>;
      chomp $line;
      @params = split /\s+/, $line;
      shift @params;
      @rm_bias=();
      push @rm_bias, @params[1..$output_dim]; 

      print OUT " [ @rm_bias ]\n"; 
#      print OUT "<!EndOfComponent>\n";
#      if($layer_cnt==3)
#      {
#        print "@rm_bias\n";
#	exit 1;
#      }
    }
    if($line=~/ComponentName/ && $line=~/backward.W_all/)
    {
      $find=1;	    
    # read ifco_x and ifco_r
      @ix=();   # $cell_dim * $input_dim
      @ir=();   # $cell_dim * $output_dim
      @fx=();
      @fr=();
      @cx=();
      @cr=();
      @ox=();
      @or=();
      $gate_cnt=0;
      while($gate_cnt < 4)
      {
        $cell_cnt = 0;
        while($cell_cnt < $cell_dim)
	{
	  $line=<IN>;
	  chomp $line;
	  @params = split /\s+/, $line;
	  if($gate_cnt == 0)
	  {
	     push @ix, @params[1..$input_dim];
	     push @ir, @params[$input_dim+1..$input_dim+$output_dim];
	  }
	  if($gate_cnt == 1)
	  {
             push @fx, @params[1..$input_dim];
             push @fr, @params[$input_dim+1..$input_dim+$output_dim];
	  }
          if($gate_cnt == 2)
          {
             push @cx, @params[1..$input_dim];
             push @cr, @params[$input_dim+1..$input_dim+$output_dim];
          }
          if($gate_cnt == 3)
          {
             push @ox, @params[1..$input_dim];
             push @or, @params[$input_dim+1..$input_dim+$output_dim];
          }
	  $cell_cnt++;
	}
	$gate_cnt++;
      }

      ########## write nne1 cifo_x ###########
#      print OUT "<LstmProjectedNnet3Streams> $output_dim $input_dim\n";
#      print OUT "<CellDim> $cell_dim <ClipGradient> 5 <DiscardInput> $discard\n";
      print OUT " [\n";
      $cnt = 0;
      while($cnt < $cell_dim)
      {
	print OUT "  @cx[$cnt*$input_dim..($cnt+1)*$input_dim-1]\n";
        $cnt++;
      }
      $cnt = 0;
      while($cnt < $cell_dim)
      {
	print OUT "  @ix[$cnt*$input_dim..($cnt+1)*$input_dim-1]\n";
	$cnt++;
      }
      $cnt = 0;
      while($cnt < $cell_dim)
      {
        print OUT "  @fx[$cnt*$input_dim..($cnt+1)*$input_dim-1]\n";
        $cnt++;
      }
      $cnt = 0;
      while($cnt < $cell_dim)
      {
	if ($cnt == ($cell_dim-1))
	{
	  print OUT "  @ox[$cnt*$input_dim..($cnt+1)*$input_dim-1] ]\n";
	  last;
	}
        print OUT "  @ox[$cnt*$input_dim..($cnt+1)*$input_dim-1]\n";
        $cnt++;
      }

      ######### write nnet1 cifo_r ##########
      print OUT " [\n";
      $cnt = 0;
      while($cnt < $cell_dim)
      {
        print OUT "  @cr[$cnt*$output_dim..($cnt+1)*$output_dim-1]\n";
        $cnt++;
      }
      $cnt = 0;
      while($cnt < $cell_dim)
      {
        print OUT "  @ir[$cnt*$output_dim..($cnt+1)*$output_dim-1]\n";
        $cnt++;
      }
      $cnt = 0;
      while($cnt < $cell_dim)
      {
        print OUT "  @fr[$cnt*$output_dim..($cnt+1)*$output_dim-1]\n";
        $cnt++;
      }
      $cnt = 0;
      while($cnt < $cell_dim)
      {
        if ($cnt == ($cell_dim-1))
        {
          print OUT "  @or[$cnt*$output_dim..($cnt+1)*$output_dim-1] ]\n";
          last;
        }
	print OUT "  @or[$cnt*$output_dim..($cnt+1)*$output_dim-1]\n";
        $cnt++;
      }

#      if ($layer_cnt==3)
#      { 
#	print "@ix[0..$input_dim-1]\n@ir[0..$output_dim-1]\n";
#	print "@ix[$input_dim*($cell_dim-1)..$input_dim*$cell_dim-1]\n@ir[$output_dim*($cell_dim-1)..$output_dim*$cell_dim-1]\n";
#        exit 1; 
#      }
         
      ######## read ifco_bias and write to nnet1 as cifo_bias #########
      $line=<IN>;
      chomp $line;
      @params = split /\s+/, $line;
      shift @params;
      @i_bias=();   # 1 * $cell_dim
      @f_bias=();
      @c_bias=();
      @o_bias=();
      push @i_bias, @params[1..$cell_dim];
      push @f_bias, @params[$cell_dim+1..$cell_dim*2];
      push @c_bias, @params[$cell_dim*2+1..$cell_dim*3];
      push @o_bias, @params[$cell_dim*3+1..$cell_dim*4];
      
      print OUT " [ @c_bias @i_bias @f_bias @o_bias ]\n";

  
#      if($layer_cnt == 3)
#      { 
#      print "@i_bias\n@f_bias\n\n@c_bias\n@o_bias\n";    
#      exit 1;
#      }

      ######### read ifo_c and write to nnet1 #######
      $line=<IN>;      #read useless line;
      $line=<IN>;      #read useless line;
    
      # read i_c;
      @i_c=();
      $line=<IN>;    
      chomp $line;
      @params = split /\s+/, $line;
      push @i_c, @params[1..$cell_dim];  
 
      # read f_c;
      @f_c=();
      $line=<IN>;
      chomp $line;
      @params = split /\s+/, $line;
      push @f_c, @params[1..$cell_dim]; 

      # read o_c;
      @o_c=();
      $line=<IN>;
      chomp $line;
      @params = split /\s+/, $line;
      push @o_c, @params[1..$cell_dim];

      print OUT " [ @i_c ]\n";
      print OUT " [ @f_c ]\n";
      print OUT " [ @o_c ]\n";
      
      # read until W_rp (r_m for nnet1)
      while($line=<IN>)
      {
        chomp $line;
        if ($line=~/ComponentName/ && $line=~/W_rp/)
        {
          last;
        }
      }
   
      ######## read W_rp and rp_bias and write to nnet1 #########
      $cnt=0;
      @r_m=();    # $output_dim * $cell_dim
      while($cnt < $output_dim)
      {
        $line=<IN>;
        chomp $line;
        @params = split /\s+/, $line;
        push @r_m, @params[1..$cell_dim];
        $cnt++;
      }
   
      print OUT " [\n";
      $cnt=0;
      while($cnt < $output_dim)
      {
        if ($cnt == ($output_dim-1))
        {
          print OUT "  @r_m[$cnt*$cell_dim..($cnt+1)*$cell_dim-1] ]\n";
          last;
        }
        print OUT "  @r_m[$cnt*$cell_dim..($cnt+1)*$cell_dim-1]\n";
        $cnt++;
      }      
#      if($layer_cnt==3)
#      {
#        print "@r_m[0..$cell_dim-1]\n@r_m[$cell_dim*($output_dim-1)..$cell_dim*$output_dim-1]\n";
#        exit 1;
#      }
      # read rm_bias
      $line=<IN>;
      chomp $line;
      @params = split /\s+/, $line;
      shift @params;
      @rm_bias=();
      push @rm_bias, @params[1..$output_dim]; 

      print OUT " [ @rm_bias ]\n"; 
#      print OUT "<!EndOfComponent>\n";
#      if($layer_cnt==3)
#      {
#        print "@rm_bias\n";
#	exit 1;
#      }
      last;
    }
	
 }
  if ($find == 0) {
    print "Error: Can't find $_[0] in nnet3 model file.\n";
    exit 1;
  }
  print "converting  $_[0] finished...\n";
}


close(IN);
close(OUT);

close(PROTO);
