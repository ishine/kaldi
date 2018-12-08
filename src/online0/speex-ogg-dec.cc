// online0/speex-ogg-dec.cc

// Copyright 2018	Alibaba Inc (author: Wei Deng)

// See ../../COPYING for clarification regarding multiple authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
// WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
// MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache 2 License for the specific language governing permissions and
// limitations under the License.


#ifdef HAVE_SPEEX

#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <cstring>
#include <vector>

#include <ogg/ogg.h>
#include <speex/speex.h>
#include <speex/speex_header.h>
#include <speex/speex_stereo.h>
#include <speex/speex_callbacks.h>

#include "online0/speex-ogg-dec.h"

#define MAX_FRAME_SIZE 2000

namespace kaldi {

static void *process_header(ogg_packet *op, spx_int32_t enh_enabled, spx_int32_t *frame_size, int *granule_frame_size, spx_int32_t *rate, int *nframes, int forceMode, int *channels, SpeexStereoState *stereo, int *extra_headers, int quiet)
{
   void *st;
   const SpeexMode *mode;
   SpeexHeader *header;
   int modeID;
   SpeexCallback callback;

   header = speex_packet_to_header((char*)op->packet, op->bytes);
   if (!header)
   {
      fprintf (stderr, "Cannot read header\n");
      return NULL;
   }
   if (header->mode >= SPEEX_NB_MODES || header->mode<0)
   {
      fprintf (stderr, "Mode number %d does not (yet/any longer) exist in this version\n",
               header->mode);
      free(header);
      return NULL;
   }

   modeID = header->mode;
   if (forceMode!=-1)
      modeID = forceMode;

   mode = speex_lib_get_mode (modeID);

   if (header->speex_version_id > 1)
   {
      fprintf (stderr, "This file was encoded with Speex bit-stream version %d, which I don't know how to decode\n", header->speex_version_id);
      free(header);
      return NULL;
   }

   if (mode->bitstream_version < header->mode_bitstream_version)
   {
      fprintf (stderr, "The file was encoded with a newer version of Speex. You need to upgrade in order to play it.\n");
      free(header);
      return NULL;
   }
   if (mode->bitstream_version > header->mode_bitstream_version)
   {
      fprintf (stderr, "The file was encoded with an older version of Speex. You would need to downgrade the version in order to play it.\n");
      free(header);
      return NULL;
   }

   st = speex_decoder_init(mode);
   if (!st)
   {
      fprintf (stderr, "Decoder initialization failed.\n");
      free(header);
      return NULL;
   }
   speex_decoder_ctl(st, SPEEX_SET_ENH, &enh_enabled);
   speex_decoder_ctl(st, SPEEX_GET_FRAME_SIZE, frame_size);
   *granule_frame_size = *frame_size;

   if (!*rate)
      *rate = header->rate;
   /* Adjust rate if --force-* options are used */
   if (forceMode!=-1)
   {
      if (header->mode < forceMode)
      {
         *rate <<= (forceMode - header->mode);
         *granule_frame_size >>= (forceMode - header->mode);
      }
      if (header->mode > forceMode)
      {
         *rate >>= (header->mode - forceMode);
         *granule_frame_size <<= (header->mode - forceMode);
      }
   }


   speex_decoder_ctl(st, SPEEX_SET_SAMPLING_RATE, rate);

   *nframes = header->frames_per_packet;

   if (*channels==-1)
      *channels = header->nb_channels;

   if (!(*channels==1))
   {
      *channels = 2;
      callback.callback_id = SPEEX_INBAND_STEREO;
      callback.func = speex_std_stereo_request_handler;
      callback.data = stereo;
      speex_decoder_ctl(st, SPEEX_SET_HANDLER, &callback);
   }

   if (!quiet)
   {
      fprintf (stderr, "Decoding %d Hz audio using %s mode",
               *rate, mode->modeName);

      if (*channels==1)
         fprintf (stderr, " (mono");
      else
         fprintf (stderr, " (stereo");

      if (header->vbr)
         fprintf (stderr, ", VBR)\n");
      else
         fprintf(stderr, ")\n");
      /*fprintf (stderr, "Decoding %d Hz audio at %d bps using %s mode\n",
       *rate, mode->bitrate, mode->modeName);*/
   }

   *extra_headers = header->extra_headers;

   free(header);
   return st;
}

int SpeexOggDecoder(const char *speex_ogg_bits, int flen, char* &audio_bits)
{
	ogg_int64_t page_granule=0, last_granule=0;
	ogg_sync_state oy;
	ogg_page       og;
	ogg_packet     op;
	ogg_stream_state os;

	SpeexBits bits;
	SpeexStereoState stereo = SPEEX_STEREO_STATE_INIT;

	short out[MAX_FRAME_SIZE];
	short output[MAX_FRAME_SIZE];
	int frame_size=0, granule_frame_size=0;
	void *st=NULL;
	int packet_count=0;
	int stream_init = 0;
	int quiet = 1;

	int skip_samples=0, page_nb_packets;


	int enh_enabled;
	int nframes=2;
	int print_bitrate=0;
	int eos=0;
	int forceMode=-1;
	int audio_size=0;
	float loss_percent=-1;
	int channels=-1;
	int rate=0;
	int extra_headers=0;
	int lookahead;
	int speex_serialno = -1;

	enh_enabled = 1;

	std::vector<short> out_buffer;
	int offset = 0;


	/*Init Ogg data struct*/
	ogg_sync_init(&oy);
	speex_bits_init(&bits);

	/*Main decoding loop*/
	while (offset < flen)
	{
		char *data;
		int i, j, nb_read;
		/*Get the ogg buffer for writing*/
		data = ogg_sync_buffer(&oy, 200);

		nb_read = flen-offset>=200 ? 200 : flen-offset;
		memcpy(data, speex_ogg_bits+offset, nb_read);
		ogg_sync_wrote(&oy, nb_read);
		offset += nb_read;

		/*Loop for all complete pages we got (most likely only one)*/
		while (ogg_sync_pageout(&oy, &og)==1)
		{
			int packet_no;
			if (stream_init == 0) {
				ogg_stream_init(&os, ogg_page_serialno(&og));
				stream_init = 1;
			}

			if (ogg_page_serialno(&og) != os.serialno) {
				/* so all streams are read. */
				ogg_stream_reset_serialno(&os, ogg_page_serialno(&og));
			}

			/*Add page to the bitstream*/
			ogg_stream_pagein(&os, &og);
			page_granule = ogg_page_granulepos(&og);
			page_nb_packets = ogg_page_packets(&og);

			if (page_granule>0 && frame_size)
			{
				/* FIXME: shift the granule values if --force-* is specified */
				skip_samples = frame_size*(page_nb_packets*granule_frame_size*nframes - (page_granule-last_granule))/granule_frame_size;
				if (ogg_page_eos(&og))
				   skip_samples = -skip_samples;
				/*else if (!ogg_page_bos(&og))
				   skip_samples = 0;*/
			} else {
				skip_samples = 0;
			}

			/*printf ("page granulepos: %d %d %d\n", skip_samples, page_nb_packets, (int)page_granule);*/
			last_granule = page_granule;

			/*Extract all available packets*/
			packet_no=0;
			while (!eos && ogg_stream_packetout(&os, &op) == 1)
			{
				if (op.bytes>=5 && !memcmp(op.packet, "Speex", 5)) {
					speex_serialno = os.serialno;
				}
				if (speex_serialno == -1 || os.serialno != speex_serialno)
					break;
				/*If first packet, process as Speex header*/
				if (packet_count==0)
				{
					st = process_header(&op, enh_enabled, &frame_size, &granule_frame_size, &rate, &nframes, forceMode, &channels, &stereo, &extra_headers, quiet);
					if (!st) exit(1);
					speex_decoder_ctl(st, SPEEX_GET_LOOKAHEAD, &lookahead);
					if (!nframes)
						nframes=1;

				} else if (packet_count==1) {
                    /*
					if (!quiet)
						print_comments((char*)op.packet, op.bytes);
                    */
				} else if (packet_count<=1+extra_headers) {
					/* Ignore extra headers */
				} else {
				   int lost=0;
				   packet_no++;
				   if (loss_percent>0 && 100*((float)rand())/RAND_MAX<loss_percent)
					  lost=1;

				   /*End of stream condition*/
				   if (op.e_o_s && os.serialno == speex_serialno) /* don't care for anything except speex eos */
					   eos=1;

					/*Copy Ogg packet to Speex bitstream*/
					speex_bits_read_from(&bits, (char*)op.packet, op.bytes);
					for (j=0; j!=nframes; j++)
					{
						int ret;
						/*Decode frame*/
						if (!lost)
						 ret = speex_decode_int(st, &bits, output);
						else
						 ret = speex_decode_int(st, NULL, output);

						/*for (i=0;i<frame_size*channels;i++)
						printf ("%d\n", (int)output[i]);*/

						if (ret==-1)
						 break;
						if (ret==-2)
						{
							fprintf (stderr, "Decoding error: corrupted stream?\n");
							break;
						}
						if (speex_bits_remaining(&bits)<0)
						{
							fprintf (stderr, "Decoding overflow: corrupted stream?\n");
							break;
						}
						if (channels==2)
							speex_decode_stereo_int(output, frame_size, &stereo);

						if (print_bitrate) {
							spx_int32_t tmp;
							char ch=13;
							speex_decoder_ctl(st, SPEEX_GET_BITRATE, &tmp);
							fputc (ch, stderr);
							fprintf (stderr, "Bitrate is use: %d bps     ", tmp);
						}

						/*Convert to short and save to buffer*/
						for (i=0;i<frame_size*channels;i++)
							out[i]=output[i];

						{
							int frame_offset = 0;
							int new_frame_size = frame_size;
							/*printf ("packet %d %d\n", packet_no, skip_samples);*/
							/*fprintf (stderr, "packet %d %d %d\n", packet_no, skip_samples, lookahead);*/
							if (packet_no == 1 && j==0 && skip_samples > 0) {
								/*printf ("chopping first packet\n");*/
								new_frame_size -= skip_samples+lookahead;
								frame_offset = skip_samples+lookahead;
							}
							if (packet_no == page_nb_packets && skip_samples < 0) {
								int packet_length = nframes*frame_size+skip_samples+lookahead;
								new_frame_size = packet_length - j*frame_size;
								if (new_frame_size<0)
								   new_frame_size = 0;
								if (new_frame_size>frame_size)
								   new_frame_size = frame_size;
								/*printf ("chopping end: %d %d %d\n", new_frame_size, packet_length, packet_no);*/
							}
							if (new_frame_size>0) {
								// fwrite(out+frame_offset*channels, sizeof(short), new_frame_size*channels, fout);
								for (int i = 0; i < new_frame_size*channels; i++)
									out_buffer.push_back(out[frame_offset*channels+i]);
								audio_size+=sizeof(short)*new_frame_size*channels;
							}
						}
					}
				}
				packet_count++;
			} //end while
		}// end page
   }

   speex_bits_destroy(&bits);
   if (stream_init)
      ogg_stream_clear(&os);
   ogg_sync_clear(&oy);

	if (st)
	   speex_decoder_destroy(st);
	else {
		fprintf (stderr, "This doesn't look like a Speex file\n");
		return 0;
	}

	int olen = out_buffer.size()*sizeof(short);
	audio_bits = new char[olen];
	memcpy(audio_bits, (char*)(&out_buffer.front()), olen);

	return olen;
}

} // namespace kaldi

#endif
