#!/usr/bin/perl -w
# This script convert kaldi's nnet3 to nnet1. It only support the lstm structure for now.
# Other nnet layers will be added later.
# Date: Thur. Oct 27 2017 -- Wangzhichao
#########################################################################################
if(@ARGV!=2)
{
  print "Usage::perl convert_nnet3_to_nnet1.pl <model-in> <model-out> \n";
  exit 1;
}

################### Set net config ######################
$lstm_layers=3;      #lstm layer number
$lstm_nodes="355 2560 854;854 2560 854;854 2560 854";   # nodes config per layer, separated by semicolons; (each layer:input_dim cell_dim output_dim)
$output_node=3766;    #the final layer i.e. output layer node number

################### Net conf end ########################
$model_in=$ARGV[0];
$model_out=$ARGV[1];

@nodes=split /;/, $lstm_nodes;
if(@nodes != $lstm_layers)
{
  print "Error: layer num and nodes config not match!\n";
  exit 1;
}

open IN, "<$model_in" or die "$!";
open OUT, ">$model_out" or die "$!";

print OUT "<Nnet>\n";

# read nnet3 lstm params layer by layer
$layer_cnt = 0;
while($layer_cnt < $lstm_layers)
{
  @current_nodes = split /\s+/, $nodes[$layer_cnt];
  $input_dim = $current_nodes[0];
  $cell_dim = $current_nodes[1];
  $output_dim = $current_nodes[2];
  
  while($line=<IN>)
  {
    chomp $line;
    if($line=~/ComponentName/ && $line=~/W_all/)
    {
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
  $layer_cnt++;
}

# read the final affine layer and write to nnet1
$line=<IN>;
$line=<IN>;
@final_affine=();   # $output_node * $output_dim;
if($line=~/ComponentName/ && $line=~/output\.affine/)
{
  $cnt=0;
  while($cnt < $output_node)
  {
    $line=<IN>;
    chomp $line;
    @params=split /\s+/, $line;
    push @final_affine, @params[1..$output_dim];
    $cnt++;
  }
}else
{
  print "Error: should be final layer but not!(Now the nnet config must be:n*lstm+final layer)\n";
}

print OUT "<AffineTransform> $output_node $output_dim\n";
print OUT "<LearnRateCoef> 2.5 <BiasLearnRateCoef> 2.5 <MaxNorm> 0\n";
print OUT " [\n";
$cnt=0;
while($cnt < $output_node)
{
  if ($cnt == ($output_node-1))
  {
    print OUT "  @final_affine[$cnt*$output_dim..($cnt+1)*$output_dim-1] ]\n";
    last;
  }
  print OUT "  @final_affine[$cnt*$output_dim..($cnt+1)*$output_dim-1]\n"; 
  $cnt++;
}

$line=<IN>;
chomp $line;
@params = split /\s+/, $line;
shift @params;
@final_bias=();   # 1 * $output_node; 
push @final_bias, @params[1..$output_node];

print OUT " [ @final_bias ]\n";
#print OUT "<!EndOfComponent>\n";
print OUT "</Nnet>\n";
#print "@final_affine[0..$output_dim-1]\n@final_affine[$output_dim*($output_node-1)..$output_dim*$output_node-1]\n";
#print "@final_bias\n";
close(IN);
close(OUT);

