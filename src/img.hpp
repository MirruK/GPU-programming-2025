/*
From the ppm documentation at https://netpbm.sourceforge.net/doc/ppm.html :
Each ppm image contains the following:
0. A "magic number" for identifying the file type. A ppm image's magic number is the two characters "P6".
1. Whitespace (blanks, TABs, CRs, LFs).
2. A width, formatted as ASCII characters in decimal.
3. Whitespace.
4. A height, again in ASCII decimal.
5. Whitespace.
6. The maximum color value (Maxval), again in ASCII decimal. Must be less than 65536 and more than zero.
7. A single whitespace character (usually a newline).
8. A raster of Height rows, in order from top to bottom. Each row consists of Width pixels, in order from left to right. Each pixel is a triplet of red, green, and blue samples, in that order. Each sample is represented in pure binary by either 1 or 2 bytes. If the Maxval is less than 256, it is 1 byte. Otherwise, it is 2 bytes. The most significant byte is first. */

typedef struct {
  unsigned short x;
  unsigned short y;
  unsigned short z;  
}PPMPixel;

voi pdarse_ppm_row(uint8_t* row) {
  
}




class PPMImage {
  int width;
  int height;
  /* Number of distict values for each color 0-65536 */
  unsigned short color_depth;
public:
  PPMImage();

  // TODO: How should construction in different ways be handled?
  static PPMImage from_random();
  
  void parse_from_file();
}
