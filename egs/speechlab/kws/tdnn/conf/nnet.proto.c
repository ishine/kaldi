<NnetProto>
<TimeDelayTransform> <InputDim> 920 <OutputDim> 3584 <NumInputContext> 23 <NumOutputContext> 7 <NumIndexes> 5 <InputContext> -2 -1 0 1 2 <Indexes> 0 1 2 3 4 <Indexes> 3 4 5 6 7 <Indexes> 6 7 8 9 10 <Indexes> 9 10 11 12 13 <Indexes> 12 13 14 15 16 <Indexes> 15 16 17 18 19 <Indexes> 18 19 20 21 22 <BiasMean> 0.05 <BiasRange> 0.1 <ParamStddev> 0.010976
<Relu> <InputDim> 3584 <OutputDim> 3584
<CompressedTimeDelayTransform> <InputDim> 3584 <OutputDim> 2048 <NumInputContext> 7 <NumOutputContext> 4 <NumIndexes> 2 <InputContext> -1 2 <Indexes> 0 1 <Indexes> 2 3 <Indexes> 3 4 <Indexes> 5 6  <BiasMean> 0.05 <BiasRange> 0.1 <ParamStddev> 0.010976 <Rank> 128
<Relu> <InputDim> 2048 <OutputDim> 2048
<CompressedTimeDelayTransform> <InputDim> 2048 <OutputDim> 1024 <NumInputContext> 4 <NumOutputContext> 2 <NumIndexes> 2 <InputContext> -3 3 <Indexes> 0 1 <Indexes> 2 3 <BiasMean> 0.05 <BiasRange> 0.1 <ParamStddev> 0.010976 <Rank> 128
<Relu> <InputDim> 1024 <OutputDim> 1024
<CompressedTimeDelayTransform> <InputDim> 1024 <OutputDim> 512 <NumInputContext> 2 <NumOutputContext> 1 <NumIndexes> 2 <InputContext> -7 2 <Indexes> 0 1 <BiasMean> 0.05 <BiasRange> 0.1 <ParamStddev> 0.010976 <Rank> 128
<Relu> <InputDim> 512 <OutputDim> 512
<AffineTransform> <InputDim> 512 <OutputDim> 411 <BiasMean> 0.02 <BiasRange> 0.05000 <ParamStddev> 0.002
<Softmax> <InputDim> 411 <OutputDim> 411
</NnetProto>
