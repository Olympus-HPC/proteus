int i4_ceiling ( double x )

/*
  Purpose:

    I4_CEILING rounds an R8 up to the nearest I4.

  Discussion:

    The "ceiling" of X is the value of X rounded towards plus infinity.

  Example:

    X        I4_CEILING(X)

   -1.1      -1
   -1.0      -1
   -0.9       0
   -0.1       0
    0.0       0
    0.1       1
    0.9       1
    1.0       1
    1.1       2
    2.9       3
    3.0       3
    3.14159   4

  Licensing:

    This code is distributed under the GNU LGPL license.

  Modified:

    10 November 2011

  Author:

    John Burkardt

  Parameters:

    Input, double X, the number whose ceiling is desired.

    Output, int I4_CEILING, the ceiling of X.
*/
{
  int value;

  value = ( int ) x;

  if ( value < x )
  {
    value = value + 1;
  }

  return value;
}

__device__
double potential ( double a, double b, double x, double y )

/*
  Purpose:

    POTENTIAL evaluates the potential function V(X,Y,Z).

  Licensing:

    This code is distributed under the GNU LGPL license. 

  Modified:

    19 February 2008

  Author:

    John Burkardt

  Parameters:

    Input, double A, B, the parameters that define the ellipse.

    Input, double X, Y, the coordinates of the point.

    Output, double POTENTIAL, the value of the potential function at (X,Y).
*/
{
  double value;

  value = 2.0 * ( pow ( x / a / a, 2.0 )
                + pow ( y / b / b, 2.0 ) )
              + 1.0 / a / a
              + 1.0 / b / b;

  return value;
}

__device__
double r8_uniform_01 ( int *seed )

/*
  Purpose:

    R8_UNIFORM_01 returns a unit pseudorandom R8.

  Discussion:

    This routine implements the recursion

      seed = 16807 * seed mod ( 2^31 - 1 )
      r8_uniform_01 = seed / ( 2^31 - 1 )

    The integer arithmetic never requires more than 32 bits,
    including a sign bit.

    If the initial seed is 12345, then the first three computations are

      Input     Output      R8_UNIFORM_01
      SEED      SEED

         12345   207482415  0.096616
     207482415  1790989824  0.833995
    1790989824  2035175616  0.947702

  Licensing:

    This code is distributed under the GNU LGPL license. 

  Modified:

    11 August 2004

  Author:

    John Burkardt

  Reference:

    Paul Bratley, Bennett Fox, Linus Schrage,
    A Guide to Simulation,
    Springer Verlag, pages 201-202, 1983.

    Pierre L'Ecuyer,
    Random Number Generation,
    in Handbook of Simulation
    edited by Jerry Banks,
    Wiley Interscience, page 95, 1998.

    Bennett Fox,
    Algorithm 647:
    Implementation and Relative Efficiency of Quasirandom
    Sequence Generators,
    ACM Transactions on Mathematical Software,
    Volume 12, Number 4, pages 362-376, 1986.

    Peter Lewis, Allen Goodman, James Miller,
    A Pseudo-Random Number Generator for the System/360,
    IBM Systems Journal,
    Volume 8, pages 136-143, 1969.

  Parameters:

    Input/output, int *SEED, the "seed" value.  Normally, this
    value should not be 0.  On output, SEED has been updated.

    Output, double R8_UNIFORM_01, a new pseudorandom variate, strictly between
    0 and 1.
*/
{
  int k;
  double r;

  k = *seed / 127773;

  *seed = 16807 * ( *seed - k * 127773 ) - k * 2836;

  if ( *seed < 0 )
  {
    *seed = *seed + 2147483647;
  }
/*
  Although SEED can be represented exactly as a 32 bit integer,
  it generally cannot be represented exactly as a 32 bit real number!
*/
  r = ( double ) ( *seed ) * 4.656612875E-10;

  return r;
}

