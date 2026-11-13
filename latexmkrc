# Make the bundled IEEE Access class and its fonts visible without copying them to the root.
$ENV{'TEXINPUTS'}   = './template//:' . ($ENV{'TEXINPUTS'}   // '');
$ENV{'TFMFONTS'}    = './template//:' . ($ENV{'TFMFONTS'}    // '');
$ENV{'T1FONTS'}     = './template//:' . ($ENV{'T1FONTS'}     // '');
$ENV{'TEXFONTMAPS'} = './template//:' . ($ENV{'TEXFONTMAPS'} // '');
$ENV{'BSTINPUTS'}   = './template//:' . ($ENV{'BSTINPUTS'}   // '');
$pdf_mode = 1;
$bibtex_use = 2;
$out_dir = 'build';
