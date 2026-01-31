1.Memory Access problmet
"Uncoalesced global access, expected 1136520 sectors, got 3409560 (3.00x) at PC  0x7fff12263b10" flera av dessa så de väll kernel requester 1byte får iställer 3
Guess solution e fix memory addresses för bättre access

2. Warp stalls
   On average, each warp of this kernel spends 9.6 cycles being stalled waiting for a scoreboard dependency on a L1TEX (local, global, surface, texture) operation. This represents about 35.8% of the total average of 26.7 cycles between issuing two instructions. To reduce the number of cycles waiting on L1TEX data accesses verify the memory access patterns are optimal for the target architecture, attempt to increase cache hit rates by increasing data locality or by changing the cache configuration, and consider moving frequently used data to shared memory.
   -> global memory loads aka ladda bilden så:
   typedef struct {
   uint16_t r;
   uint16_t g;
   uint16_t b;  
   }PPMPixel;

de e 6 bytes så add bara padding av 2 bytes o iguess

så typ:

typedef struct {
uint16_t r;
uint16_t g;
uint16_t b;  
uint_16 padding:
}PPMPixel;

thread 0= 0-7 thred 1 = 8-15 osv följer få väll cache linjerna
