#include "img.hpp"
#include <cstdint>
#include <cstdio>
#include <vector>
#include <string>
#include <stdexcept>
#include <algorithm>

PPMImage::PPMImage(){}

void consume_whitespace(std::FILE* fp) {
  char c;
  while(isspace((c =std::fgetc(fp))));
  // Avoid consuming first non-whitespace char
  std::ungetc(c, fp);
}

int parse_digit(std::FILE* fp){
  char c;
  std::string buf;
  while(isdigit(c = std::fgetc(fp))) buf.push_back(c);
  return std::stoi(buf);
}

PPMPixel parse_pixel_binary(std::FILE* fp){
  auto r = static_cast<uint16_t>(std::fgetc(fp));
  auto g = static_cast<uint16_t>(std::fgetc(fp));
  auto b = static_cast<uint16_t>(std::fgetc(fp)); 
  return {r,g,b};
}

PPMPixel parse_pixel_ascii(std::FILE* fp){
  consume_whitespace(fp);
  auto r = static_cast<uint16_t>(parse_digit(fp));
  consume_whitespace(fp);
  auto g = static_cast<uint16_t>(parse_digit(fp));
  consume_whitespace(fp);
  auto b = static_cast<uint16_t>(parse_digit(fp)); 
  return {r,g,b};
}

PPMPixel parse_pixel_binary_wide(std::FILE* fp){
  uint16_t v = 0;
  // place most significant bits into lower part
  v = std::fgetc(fp);
  // Shift them to the most significant part
  v <<= 8;
  // Place the least significant bits in their place
  v |= std::fgetc(fp);
  auto r = v;
  v = std::fgetc(fp);
  v <<= 8;
  v |= std::fgetc(fp);
  auto g = v;
  v = std::fgetc(fp);
  v <<= 8;
  v |= std::fgetc(fp);
  auto b = v; 
  return {r,g,b};
}

PPMPixel parse_pixel_ascii_wide(std::FILE* fp){
  // TODO: Implement
  return {0,0,0};
}

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


PPMImage PPMImage::from_file(std::FILE* fp) {
  PPMImage img;
  std::string buf;
  // 0. Find magic number
  buf.push_back(fgetc(fp));
  buf.push_back(fgetc(fp));
  // TODO: Support P1 (grayscale) and P3 (non-binary fmt)
  if (buf != "P6" && buf != "P3") {
    throw std::invalid_argument("Invalid file format for parsing into PPM");
  }
  PPMPixel (*parse_pixel)(std::FILE*);
  // 1. Skip whitespace
  consume_whitespace(fp);
  // 2. Parse image dimensions
  img.width = parse_digit(fp);
  consume_whitespace(fp);
  img.height = parse_digit(fp);
  consume_whitespace(fp);
  img.color_depth = parse_digit(fp);
  consume_whitespace(fp);
  // 3. allocate space for image
  img.pixels = std::vector<PPMPixel>(img.width * img.height);
  // Select pixel parsing function (Horrendous if-statement, I know)
  // Yeye polymorphism bla bla use an interface etc... fix it later
  if (buf == "P6") {
    if (img.color_depth < 256) {
      parse_pixel = &parse_pixel_binary;
    } else {
      parse_pixel = &parse_pixel_binary_wide;
    }
  }
  else {
    if (img.color_depth < 256) {
      parse_pixel = &parse_pixel_ascii;
    } else {
      parse_pixel = &parse_pixel_ascii_wide;
    }
  }
  // 4. Parse height rows of width pixels
  for(int i = 0; i < img.height; i++){
    for(int j = 0; j < img.width; j++){
	img.pixels[i*img.width + j] = parse_pixel(fp);
    }
  }
  return img;
}


void PPMImage::to_file(std::FILE* fp) {
  // 1. Write magic sequence PM6
  fprintf(fp,"P6\n");
  // 2. Print height, width and color_depth. Finally a newline
  fprintf(fp, "%d %d\n%d\n", width, height, color_depth);
  // 3. Pixel data as height number of width-wide rows
  PPMPixel px;
  for(int i = 0; i < height; i++){
    for(int j = 0; j < width; j++){
      if (color_depth < 256) {
	px = pixels[i*width + j];
	fputc(static_cast<uint8_t>(px.r),fp);
	fputc(static_cast<uint8_t>(px.g),fp);
	fputc(static_cast<uint8_t>(px.b),fp);
      } else{
	px = pixels[i*width + j];
	fputc(px.r >> 8,fp);
	fputc(px.r & (uint16_t)255,fp);
	fputc(px.g >> 8,fp);
	fputc(px.g & (uint16_t)255,fp);
	fputc(px.b >> 8,fp);
	fputc(px.b & (uint16_t)255,fp);
      }
    }
  }
}
