class Sarray {
  public:
    Sarray() {}
    ~Sarray();
    Sarray(int nc, int ibeg, int iend, int jbeg, int jend, int kbeg, int kend);
    std::string fill(std::istringstream& iss);
    void init();
    float_sw4 norm();
    std::tuple<float_sw4,float_sw4> minmax();
    int m_nc, m_ni, m_nj, m_nk;
    int m_ib, m_ie, m_jb, m_je, m_kb, m_ke;
    ssize_t m_base;
    size_t m_offi, m_offj, m_offk, m_offc, m_npts;
    float_sw4* m_data;
    size_t size;
    int g;
};

std::string Sarray::fill(std::istringstream& iss) {
  std::string name;
  if (!(iss >> name >> g >> m_nc >> m_ni >> m_nj >> m_nk >> m_ib >> m_ie >>
        m_jb >> m_je >> m_kb >> m_ke >> m_base >> m_offi >> m_offj >> m_offk >>
        m_offc >> m_npts))
    return "Break";
#ifdef VERBOSE
  std::cout << name << " " << m_npts << "\n";
#endif
  size = m_nc * m_ni * m_nj * m_nk * sizeof(float_sw4);

  float_sw4* ptr = (float_sw4*) malloc (size);
  if (ptr == nullptr) {
    std::cerr << "malloc failed (size:" << size << " bytes)\n";
    abort();
  }

#ifdef VERBOSE
  std::cout << "Allocated " << size << " bytes " << name << "[" << g << "]\n";
#endif
  m_data = ptr;
  return name;
}

Sarray::~Sarray() {
#ifdef VERBOSE
  std::cout << "Free " << size << " bytes\n";
#endif
  free(m_data);
}

void Sarray::init() {

  const float_sw4 dx = 0.001;
  int nc = m_nc;
  int offi = nc;
  int offj = nc*m_ni;
  int offk = nc*m_ni*m_nj;

  for (int i = 0; i < m_ni; i++)
    for (int j = 0; j < m_nj; j++)
      for (int k = 0; k < m_nk; k++)
        for (int c = 0; c < nc; c++) {
          int indx = c + i * offi + j * offj + k * offk;
          float_sw4 x = i*dx;
          float_sw4 y = j*dx;
          float_sw4 z = k*dx;
          float_sw4 f = sin(x)*sin(y)*sin(z);
          m_data[indx]=f;
        }

}
float_sw4 Sarray::norm() {
  float_sw4 ret = 0.0;
  for (size_t i = 0; i < size / 8; i++) ret += m_data[i] * m_data[i];
  return ret;
}

std::tuple<float_sw4,float_sw4> Sarray::minmax(){
  float_sw4 min = std::numeric_limits<float_sw4>::max();
  float_sw4 max = std::numeric_limits<float_sw4>::min();
  for (size_t i = 0; i < size / 8; i++) {
    min=std::min(min,m_data[i]);
    max=std::max(max,m_data[i]);
  }
  return std::make_tuple(min,max);
}


void CheckError(cudaError_t const err, const char *file, char const *const fun,
    const int line) {
  if (err) {
    std::cerr << "CUDA Error Code[" << err << "]: " << cudaGetErrorString(err)
      << " " << file << " " << fun << " Line number:  " << line << "\n";
    abort();
  }
}

#define CheckDeviceError(err) \
  CheckError(err, __FILE__, __FUNCTION__, __LINE__)


