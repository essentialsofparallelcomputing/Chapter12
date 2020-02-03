eval 'exec perl -w $0 ${1+"$@"}'
  if 0;

use File::Basename;

$ARGV_LAST = "";

($file,,) = fileparse($ARGV[0], qr/\..*/);
print "const char *",$file,"_source =\n";

while(<>) {
   if ($ARGV ne $ARGV_LAST){
      $ARGV_LAST=$ARGV;
   }
   chop $_;

   s/"/""/g;
   s/\\/\\\\/g;

   print '"',$_,'\n','"',"\n";
}

print ";"
