/**
 * Project  : ACT(A.I.Labs C/CPP Toolkits) -- tools.v1.0.1
 * FileName : src/tools/asr-decoder.cpp
 *
 * COPYRIGHT (C) 2018, A.I.Labs Group. All rights reserved.
 */
#include "vad_api.h"
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include <signal.h>
#include <execinfo.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>

int vad_func(void *usr_ptr, int status)
{
    std::cout << "vad status:" << status << std::endl;
}

int feature_func(void *usr_ptr, float* feature, int feature_len)
{

    std::cout << "feature:";
    for(int i=0;i<feature_len;i++)
    {
        std::cout << feature[i] << ","; 
    }
    std::cout << std::endl;

}
VAD_IMPORT_OR_EXPORT struct api_vad * VAD_CALL vad_new(const char *cfg_fn);
VAD_IMPORT_OR_EXPORT int VAD_CALL vad_delete(struct api_vad *engine);
VAD_IMPORT_OR_EXPORT int VAD_CALL vad_start(struct api_vad *engine, void *usr_ptr, VADResultHandler vad_func,FeatureResultHandler feature_func);
VAD_IMPORT_OR_EXPORT int VAD_CALL vad_feed(struct api_vad *engine, const char *data, int size);
VAD_IMPORT_OR_EXPORT int VAD_CALL vad_stop(struct api_vad *engine);
VAD_IMPORT_OR_EXPORT int VAD_CALL vad_reset(struct api_vad *engine);

int main(int argc, const char *argv[]) {
    int ret=0;
    struct api_vad *ptr =vad_new(argv[1]);
    vad_start(ptr,NULL,&vad_func,&feature_func);

    FILE *pf_wav = fopen(argv[2], "rb");
    if(NULL == pf_wav)
    {
        return -1;
    }

    int readBytes = 0;
    char buf[3000];
    while (1) {
        readBytes = fread(buf, 1, 3000, pf_wav);
        if (readBytes < 0) 
        {
                printf("ERROR: read audio data error.\n");
                return -1;
        } 
        else if (readBytes == 0) 
        {
            break;
        } 
        else 
        {
            ret = vad_feed(ptr,buf, readBytes);
        }
    }
    fclose(pf_wav);
    return 1;
}


