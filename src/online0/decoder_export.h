#ifndef __YAE_DECODER_H__
#define __YAE_DECODER_H__

#include "online0/online-fst-decoder.h"

extern "C" void * CreateDecoder(const char * cfgPath);
extern "C" int DecoderFeedData(void * lpDecoder, void * data, int nbytes, int state);
extern "C" int GetResult(void * lpDecoder, int * result, int state);
extern "C" void ResetDecoder(void * lpDecoder);
extern "C" void DisposeDecoder(void ** lpDecoder);

#endif
