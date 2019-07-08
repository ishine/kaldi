if (@ARGV != 3)
{
    print "perl $0 <final.mdl.txt> <phones> <state>\n";
    exit(1);
}
($in, $phones, $out) = @ARGV;
open(IN, $phones);
while ($line = <IN>)
{
    $line =~ s/\r//g;
    $line =~ s/\n//g;
    @tmp = split(/ /, $line);
    $valid{$tmp[1]} = $tmp[0];
}
close IN;
open(IN, $in);
open(OUT, ">$out");
$line = <IN>;
while ($line !~ /Tuples/)
{
    $line = <IN>;
}
$line =~ s/\r//g;
$line =~ s/\n//g;
@tmp = split(/ /, $line);
$count = $tmp[1];
for ($i=0;$i<$count;$i++)
{
    $line = <IN>;
    $line =~ s/\r//g;
    $line =~ s/\n//g;
    @tmp = split(/ /, $line);
    $ph = $tmp[0];
    $prepdf = $tmp[2];
    $selfpdf = $tmp[3];
    if (!defined($mapr{$prepdf}))
    {
        $mapr{$prepdf} = $ph;
    }
    else
    {
        if ($mapr{$prepdf} ne $ph)
        {
            print "error 1 in $line\n";
        }
    }
    if (!defined($stater{$prepdf}))
    {
        $stater{$prepdf} = 2;
    }
    else
    {
        if ($stater{$prepdf} != 2)
        {
            $stater{$prepdf} = 2;
            print "error 2 in $line\n";
        }
    }
    if (!defined($mapr{$selfpdf}))
    {
        $mapr{$selfpdf} = $ph;
    }
    else
    {
        if ($mapr{$selfpdf} ne $ph)
        {
            print "error 3 in $line\n";
        }
    }
    if (!defined($stater{$selfpdf}))
    {
        $stater{$selfpdf} = 3;
    }
    else
    {
        if ($stater{$selfpdf} != 3)
        {
            $stater{$selfpdf} = 2;
            print "error 4 in $line\n";
        }
    }
    if (!defined($outputdone{$prepdf}))
    {
        $outprefix = $valid{$ph}."_s$stater{$prepdf}"."_";
        $outcount{$outprefix}++;
        print OUT $outprefix.$outcount{$outprefix}."\t$prepdf\n";
        $outputdone{$prepdf} = 1;
    }
    if (!defined($outputdone{$selfpdf}))
    {
        $outprefix = $valid{$ph}."_s$stater{$selfpdf}"."_";
        $outcount{$outprefix}++;
        print OUT $outprefix.$outcount{$outprefix}."\t$selfpdf\n";
        $outputdone{$selfpdf} = 1;
    }
}
close IN;
close OUT;
