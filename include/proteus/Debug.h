#ifndef PROTEUS_DEBUG_H
#define PROTEUS_DEBUG_H

#if PROTEUS_ENABLE_DEBUG
#define PROTEUS_DBG(x) x;
#else
#define PROTEUS_DBG(x)
#endif

#endif
