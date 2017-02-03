/*							p1evl()
 *                                          N
 * Evaluate polynomial when coefficient of x  is 1.0.
 * Otherwise same as polevl.
 */

double p1evl( x, coef, N )
     double x;
     double coef[];
     int N;
{
  double ans;
  double *p;
  int i;

  p = coef;
  ans = x + *p++;
  i = N-1;

  do
    ans = ans * x  + *p++;
  while( --i );

  return( ans );
}

double polevl( x, coef, N )
     double x;
     double coef[];
     int N;
{
  double ans;
  int i;
  double *p;

  p = coef;
  ans = *p++;
  i = N;

  do
    ans = ans * x  +  *p++;
  while( --i );

  return( ans );
}

double erfce (double x)
{
    double p,q;

    double P[] = {
      2.46196981473530512524E-10,
      5.64189564831068821977E-1,
      7.46321056442269912687E0,
      4.86371970985681366614E1,
      1.96520832956077098242E2,
      5.26445194995477358631E2,
      9.34528527171957607540E2,
      1.02755188689515710272E3,
      5.57535335369399327526E2
    };

    double Q[] = {
      /* 1.00000000000000000000E0,*/
      1.32281951154744992508E1,
      8.67072140885989742329E1,
      3.54937778887819891062E2,
      9.75708501743205489753E2,
      1.82390916687909736289E3,
      2.24633760818710981792E3,
      1.65666309194161350182E3,
      5.57535340817727675546E2
    };
    double R[] = {
      5.64189583547755073984E-1,
      1.27536670759978104416E0,
      5.01905042251180477414E0,
      6.16021097993053585195E0,
      7.40974269950448939160E0,
      2.97886665372100240670E0
    };
    double S[] = {
      /* 1.00000000000000000000E0,*/
      2.26052863220117276590E0,
      9.39603524938001434673E0,
      1.20489539808096656605E1,
      1.70814450747565897222E1,
      9.60896809063285878198E0,
      3.36907645100081516050E0
    };
    if (x < 8.0) {
      p = polevl(x, P, 8);
      q = p1evl(x, Q, 8);
    } else {
      p = polevl(x, R, 5);
      q = p1evl(x, S, 6);
    }

    return p/q;
}

static double myerf(double a);
static double myerfc(double a);
/* erfc function */
   
static double myerfc(a)
     double a;
{

    double P[] = {
      2.46196981473530512524E-10,
      5.64189564831068821977E-1,
      7.46321056442269912687E0,
      4.86371970985681366614E1,
      1.96520832956077098242E2,
      5.26445194995477358631E2,
      9.34528527171957607540E2,
      1.02755188689515710272E3,
      5.57535335369399327526E2
    };

    double Q[] = {
      /* 1.00000000000000000000E0,*/
      1.32281951154744992508E1,
      8.67072140885989742329E1,
      3.54937778887819891062E2,
      9.75708501743205489753E2,
      1.82390916687909736289E3,
      2.24633760818710981792E3,
      1.65666309194161350182E3,
      5.57535340817727675546E2
    };
    double R[] = {
      5.64189583547755073984E-1,
      1.27536670759978104416E0,
      5.01905042251180477414E0,
      6.16021097993053585195E0,
      7.40974269950448939160E0,
      2.97886665372100240670E0
    };
    double S[] = {
      /* 1.00000000000000000000E0,*/
      2.26052863220117276590E0,
      9.39603524938001434673E0,
      1.20489539808096656605E1,
      1.70814450747565897222E1,
      9.60896809063285878198E0,
      3.36907645100081516050E0
    };

  double p,q,x,y,z;
double MAXLOG =  7.09782712893383996843E2;


  if( a < 0.0 )
    x = -a;
  else
    x = a;

  if( x < 1.0 )
    return( 1.0 - myerf(a) );

  z = -a * a;

  if( z < -MAXLOG )
    {
    under:
	
      if( a < 0 )
	return( 2.0 );
      else
	return( 0.0 );
    }

  z = exp(z);

  if( x < 8.0 )
    {
      p = polevl( x, P, 8 );
      q = p1evl( x, Q, 8 );
    }
  else
    {
      p = polevl( x, R, 5 );
      q = p1evl( x, S, 6 );
    }
  y = (z * p)/q;

  if( a < 0 )
    y = 2.0 - y;

  if( y == 0.0 )
    goto under;

  return(y);
}

/* erf function */

static double myerf(x)
     double x;
{
	double T[] = {
      9.60497373987051638749E0,
      9.00260197203842689217E1,
      2.23200534594684319226E3,
      7.00332514112805075473E3,
      5.55923013010394962768E4
    };
    double U[] = {
      /* 1.00000000000000000000E0,*/
      3.35617141647503099647E1,
      5.21357949780152679795E2,
      4.59432382970980127987E3,
      2.26290000613890934246E4,
      4.92673942608635921086E4
    };

  double y, z;

  /* if( fabs(x) > 1.0 ) */
  /*   return( 1.0 - myerfc(x) ); */
  z = x * x;
  y = x * polevl( z, T, 4 ) / p1evl( z, U, 5 );
  return( y );

}


__kernel void c_matrix_irf (
    const unsigned int size_rates,
    const unsigned int size_times,
    const unsigned int nr_gaussian,
    const double scale,
    __global const double *rates,
    __global const double *times,
    __global const double *center,
    __global const double *width,
    __global double *res_g)
{
    int n_x = get_global_id(0);
    int n_t = get_global_id(1);
    int n_r = get_global_id(2);
    int pos = size_times * size_rates *n_x + size_rates * n_t + n_r;
	double k = rates[n_r];
	double t = times[n_t];
        double c = center[n_x*nr_gaussian ];
        double w = width[n_x*nr_gaussian ];
        double alpha = -k * w / sqrt(2.0);
        double beta = (t - c) / (w * sqrt(2.0));
        double thresh = beta - alpha;
        if (thresh < -1.0){
            res_g[pos] = scale * .5 * erfce(-thresh) * exp(-beta * beta);
        } else if (fabs(thresh) > 1.0) {
            res_g[pos] = scale * .5 * (2.0 - myerfc(thresh)) * exp(alpha * (alpha - 2 * beta));
        } else {
            res_g[pos] = scale * .5 * (1.0 + myerf(thresh)) * exp(alpha * (alpha - 2 * beta));
        }
    
}
