/* --------------------------------------------------------- */
/* --- File: cmaes.h ----------- Author: Nikolaus Hansen --- */
/* ---------------------- last modified: IX 2010         --- */
/* --------------------------------- by: Nikolaus Hansen --- */
/* --------------------------------------------------------- */
/*   
     CMA-ES for non-linear function minimization. 

     Copyright (C) 1996, 2003-2010  Nikolaus Hansen. 
     e-mail: nikolaus.hansen (you know what) inria.fr
      
     License: see file cmaes.c
   
*/
#ifndef NH_cmaes_h /* only include ones */ 
#define NH_cmaes_h 

#include <time.h>
#include <boost/thread.hpp>

namespace cmaes {

struct cmaes_random_t
/* cmaes_random_t 
 * sets up a pseudo random number generator instance 
 */
{
  /* Variables for Uniform() */
  long int startseed;
  long int aktseed;
  long int aktrand;
  long int *rgrand;
  
  /* Variables for Gauss() */
  short flgstored;
  double hold;
} ;

struct cmaes_timings_t
/* cmaes_timings_t 
 * time measurement, used to time eigendecomposition 
 */
{
  /* for outside use */
  double totaltime; /* zeroed by calling re-calling cmaes_timings_start */
  double totaltotaltime;
  double tictoctime; 
  double lasttictoctime;
  
  /* local fields */
  clock_t lastclock;
  time_t lasttime;
  clock_t ticclock;
  time_t tictime;
  short istic;
  short isstarted; 

  double lastdiff;
  double tictoczwischensumme;
};

struct cmaes_readpara_t
/* cmaes_readpara_t
 * collects all parameters, in particular those that are read from 
 * a file before to start. This should split in future? 
 */
{
  char * filename;  /* keep record of the file that was taken to read parameters */
  short flgsupplemented; 
  
  /* input parameters */
  int N; /* problem dimension, must stay constant, should be unsigned or long? */
  unsigned int seed; 
  double * xstart; 
  double * typicalX; 
  int typicalXcase;
  double * rgInitialStds;
  double * rgDiffMinChange; 

  /* termination parameters */
  double stopMaxFunEvals; 
  double facmaxeval;
  double stopMaxIter; 
  struct { int flg; double val; } stStopFitness; 
  double stopTolFun;
  double stopTolFunHist;
  double stopTolX;
  double stopTolUpXFactor;

  /* internal evolution strategy parameters */
  int lambda;          /* -> mu, <- N */
  int mu;              /* -> weights, (lambda) */
  double mucov, mueff; /* <- weights */
  double *weights;     /* <- mu, -> mueff, mucov, ccov */
  double damps;        /* <- cs, maxeval, lambda */
  double cs;           /* -> damps, <- N */
  double ccumcov;      /* <- N */
  double ccov;         /* <- mucov, <- N */
  double diagonalCov;  /* number of initial iterations */
  struct { int flgalways; double modulo; double maxtime; } updateCmode;
  double facupdateCmode;

  /* supplementary variables */

  char *weigkey; 
  char resumefile[99];
  const char **rgsformat;
  void **rgpadr;
  const char **rgskeyar;
  double ***rgp2adr;
  int n1para, n1outpara;
  int n2para;
};

struct cmaes_t
/* cmaes_t 
 * CMA-ES "object" 
 */
{
  const char *version;
  /* char *signalsFilename; */
  cmaes_readpara_t sp;
  cmaes_random_t rand; /* random number generator */

  double sigma;  /* step size */

  double *rgxmean;  /* mean x vector, "parent" */
  double *rgxbestever; 
  double **rgrgx;   /* range of x-vectors, lambda offspring */
  int *index;       /* sorting index of sample pop. */
  double *arFuncValueHist;

  short flgIniphase; /* not really in use anymore */
  short flgStop; 

  double chiN; 
  double **C;  /* lower triangular matrix: i>=j for C[i][j] */
  double **B;  /* matrix with normalize eigenvectors in columns */
  double *rgD; /* axis lengths */

  double *rgpc;
  double *rgps;
  double *rgxold; 
  double *rgout; 
  double *rgBDz;   /* for B*D*z */
  double *rgdTmp;  /* temporary (random) vector used in different places */
  double *rgFuncValue; 
  double *publicFitness; /* returned by cmaes_init() */

  double gen; /* Generation number */
  double countevals;
  double state; /* 1 == sampled, 2 == not in use anymore, 3 == updated */

  double maxdiagC; /* repeatedly used for output */
  double mindiagC;
  double maxEW;
  double minEW;

  char sOutString[330]; /* 4x80 */

  short flgEigensysIsUptodate;
  short flgCheckEigen; /* control via cmaes_signals.par */
  double genOfEigensysUpdate; 
  cmaes_timings_t eigenTimings;
 
  double dMaxSignifKond; 				     
  double dLastMinEWgroesserNull;

  short flgresumedone; 

  time_t printtime; 
  time_t writetime; /* ideally should keep track for each output file */
  time_t firstwritetime;
  time_t firstprinttime; 

};


/* --- initialization, constructors, destructors --- */
double * cmaes_init(cmaes_t *, int dimension , double *xstart,
		double *stddev, long seed, int lambda,
		const char *input_parameter_filename);
void cmaes_init_para(cmaes_t *, int dimension , double *xstart,
		double *stddev, long seed, int lambda,
		const char *input_parameter_filename);
double * cmaes_init_final(cmaes_t *);
void cmaes_resume_distribution(cmaes_t *evo_ptr, char *filename);
void cmaes_exit(cmaes_t *);

/* --- core functions --- */
double * const * cmaes_SamplePopulation(cmaes_t *);
double *         cmaes_UpdateDistribution(cmaes_t *,
					  const double *rgFitnessValues);
const char *     cmaes_TestForTermination(cmaes_t *);

/* --- additional functions --- */
double * const * cmaes_ReSampleSingle(cmaes_t *t, int index);
double const *   cmaes_ReSampleSingle_old(cmaes_t *, double *rgx);
double *         cmaes_SampleSingleInto( cmaes_t *t, double *rgx);
void             cmaes_UpdateEigensystem(cmaes_t *, int flgforce);

/* --- getter functions --- */
double         cmaes_Get(cmaes_t *, char const *keyword);
const double * cmaes_GetPtr(cmaes_t *, char const *keyword); /* e.g. "xbestever" */
double *       cmaes_GetNew( cmaes_t *t, char const *keyword); /* user is responsible to free */
double *       cmaes_GetInto( cmaes_t *t, char const *keyword, double *mem); /* allocs if mem==NULL, user is responsible to free */

/* --- online control and output --- */
void           cmaes_ReadSignals(cmaes_t *, char const *filename);
void           cmaes_WriteToFile(cmaes_t *, const char *szKeyWord,
                                 const char *output_filename);
char *         cmaes_SayHello(cmaes_t *);
/* --- misc --- */
double *       cmaes_NewDouble(int n); /* user is responsible to free */
void           cmaes_FATAL(char const *s1, char const *s2, char const *s3,
			   char const *s4);

long   cmaes_random_init(cmaes_random_t *, long unsigned seed /* 0==clock */);
void   cmaes_random_exit(cmaes_random_t *);
double cmaes_random_Gauss(cmaes_random_t *); /* (0,1)-normally distributed */
double cmaes_random_Uniform(cmaes_random_t *);
long   cmaes_random_Start(cmaes_random_t *, long unsigned seed /* 0==1 */);

void   cmaes_timings_init(cmaes_timings_t *timing);
void   cmaes_timings_start(cmaes_timings_t *timing); /* fields totaltime and tictoctime */
double cmaes_timings_update(cmaes_timings_t *timing);
void   cmaes_timings_tic(cmaes_timings_t *timing);
double cmaes_timings_toc(cmaes_timings_t *timing);

void cmaes_readpara_init (cmaes_readpara_t *, int dim, const double * xstart,
                    const double * sigma, int seed, int lambda,
                    const char * filename);
void cmaes_readpara_exit(cmaes_readpara_t *);
void cmaes_readpara_ReadFromFile(cmaes_readpara_t *, const char *szFileName);
void cmaes_readpara_SupplementDefaults(cmaes_readpara_t *);
void cmaes_readpara_SetWeights(cmaes_readpara_t *, const char * mode);
void cmaes_readpara_WriteToFile(cmaes_readpara_t *, const char *filenamedest);

const double * cmaes_Optimize( cmaes_t *, double(*pFun)(double const *, int dim),
                                long iterations);
double const * cmaes_SetMean(cmaes_t *, const double *xmean);
double * cmaes_PerturbSolutionInto(cmaes_t *t, double *xout,
                                   double const *xin, double eps);
void cmaes_WriteToFile(cmaes_t *, const char *key, const char *name);
void cmaes_WriteToFileAW(cmaes_t *t, const char *key, const char *name,
                         const char * append);
void cmaes_WriteToFilePtr(cmaes_t *, const char *key, FILE *fp);
void cmaes_ReadFromFilePtr(cmaes_t *, FILE *fp);
void cmaes_FATAL(char const *s1, char const *s2,
                 char const *s3, char const *s4);


/* ------------------- Locally visibly ----------------------- */

char * getTimeStr(void);
void TestMinStdDevs( cmaes_t *);
/* void WriteMaxErrorInfo( cmaes_t *); */

void Eigen( int N,  double **C, double *diag, double **Q,
                   double *rgtmp);
int  Check_Eigen( int N,  double **C, double *diag, double **Q);
void QLalgo2 (int n, double *d, double *e, double **V);
void Householder2(int n, double **V, double *d, double *e);
void Adapt_C2(cmaes_t *t, int hsig);

void FATAL(char const *sz1, char const *s2,
                  char const *s3, char const *s4);
void ERRORMESSAGE(char const *sz1, char const *s2,
                         char const *s3, char const *s4);
int isNoneStr(const char * filename);
void   Sorted_index( const double *rgFunVal, int *index, int n);
int    SignOfDiff( const void *d1, const void * d2);
double douSquare(double);
double rgdouMax( const double *rgd, int len);
double rgdouMin( const double *rgd, int len);
double douMax( double d1, double d2);
double douMin( double d1, double d2);
int    intMin( int i, int j);
int    MaxIdx( const double *rgd, int len);
int    MinIdx( const double *rgd, int len);
double myhypot(double a, double b);
double * new_double( int n);
void * new_void( int n, size_t size);
char * new_string( const char *);
void assign_string( char **, const char*);


static boost::mutex readMutex;

}

#endif 
