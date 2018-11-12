#!/usr/bin/perl -w
# This script convert kaldi's nnet3 to nnet1. It only support the lstm structure for now.
# Other nnet layers will be added later.
# Date: Thur. May 31 2018 -- wangzhichao214232@sogou-inc.com
#########################################################################################
if(@ARGV!=2)
{
  print "Usage::perl convert_nnet3_to_nnet1.pl <model-in> <model-out> \n";
  exit 1;
}

################### Set net config ######################
#$NumComponents=4;
$NumComponents=31;
################### tdnn-lstm example ###################
system("cat <<EOF > nnet.proto
name=tdnn1 type=NaturalGradientAffineComponent input=355 output=1024
name=relu1 type=RectifiedLinearComponent 
name=renorm1 type=NormalizeComponent
name=splice1 type=Splice offset=[-1,0,1] input=1024 output=3072
name=tdnn2 type=NaturalGradientAffineComponent input=3072 output=1024
name=relu2 type=RectifiedLinearComponent
name=renorm2 type=NormalizeComponent
name=splice2 type=Splice offset=[-1,0,1] input=1024 output=3072
name=tdnn3 type=NaturalGradientAffineComponent input=3072 output=1024
name=relu3 type=RectifiedLinearComponent
name=renorm3 type=NormalizeComponent
name=lstm1 type=Lstm input=1024 cell=2048 output=768 discard=4
name=splice3 type=Splice offset=[-1,0,1] input=768 output=2304
name=tdnn4 type=NaturalGradientAffineComponent input=2304 output=1024
name=relu4 type=RectifiedLinearComponent
name=renorm4 type=NormalizeComponent
name=splice4 type=Splice offset=[-1,0,1] input=1024 output=3072
name=tdnn5 type=NaturalGradientAffineComponent input=3072 output=1024
name=relu5 type=RectifiedLinearComponent
name=renorm5 type=NormalizeComponent
name=lstm2 type=Lstm input=1024 cell=2048 output=512 discard=8
name=splice5 type=Splice offset=[-1,0,1] input=512 output=1536
name=tdnn6 type=NaturalGradientAffineComponent input=1536 output=1024
name=relu6 type=RectifiedLinearComponent
name=renorm6 type=NormalizeComponent
name=splice6 type=Splice offset=[-1,0,1] input=1024 output=3072
name=tdnn7 type=NaturalGradientAffineComponent input=3072 output=1024
name=relu7 type=RectifiedLinearComponent
name=renorm7 type=NormalizeComponent
name=lstm3 type=Lstm input=1024 cell=2048 output=512 discard=12
name=output type=NaturalGradientAffineComponent input=512 output=3766
EOF");

################### lstm example #######################
system("cat <<EOF > nnet.proto
name=lstm1 type=Lstm input=355 cell=2560 output=768 discard=0
name=lstm2 type=Lstm input=768 cell=2560 output=768 discard=0
name=lstm3 type=Lstm input=768 cell=2560 output=768 discard=0
name=output type=NaturalGradientAffineComponent input=768 output=3766
EOF");
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
  if($component=~/AffineComponent/) {
    &parse_affine($component);
  }
  elsif($component=~/RectifiedLinearComponent/) {
    &parse_relu($component);
  }  
  elsif($component=~/NormalizeComponen/) {
    &parse_renorm($component);
  }
  elsif($component=~/Splice/) {
    &parse_splice($component);
  }
  elsif($component=~/Lstm/) {
    &parse_lstm($component);
  }else {
    print "Error: $layer_cnt+1 th Component no support - $component\n";
    exit 1;
  }
  $layer_cnt++;
}

print OUT "</Nnet>\n";
print "Success! Converting finished!\n";

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

      if ($layer_cnt==3)
      { 
	print "@ix[0..$input_dim-1]\n@ir[0..$output_dim-1]\n";
	print "@ix[$input_dim*($cell_dim-1)..$input_dim*$cell_dim-1]\n@ir[$output_dim*($cell_dim-1)..$output_dim*$cell_dim-1]\n";
        exit 1; 
      }
         
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

  
      if($layer_cnt == 3)
      { 
      print "@i_bias\n@f_bias\n\n@c_bias\n@o_bias\n";    
      exit 1;
      }

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
      if($layer_cnt==3)
      {
        print "@r_m[0..$cell_dim-1]\n@r_m[$cell_dim*($output_dim-1)..$cell_dim*$output_dim-1]\n";
        exit 1;
      }
      # read rm_bias
      $line=<IN>;
      chomp $line;
      @params = split /\s+/, $line;
      shift @params;
      @rm_bias=();
      push @rm_bias, @params[1..$output_dim]; 

      print OUT " [ @rm_bias ]\n"; 
#      print OUT "<!EndOfComponent>\n";
      if($layer_cnt==3)
      {
        print "@rm_bias\n";
	exit 1;
      }
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
