set yrange [0.5:1.0]
set xrange [0:15]
set terminal postscript eps enhanced color font 'Helvetica,26'
set output 'username.eps'
set xlabel "Number of added tweets (thousands)"
set ylabel "Accuracy"
set key Left top left reverse

plot "BASELINE_SAME" every 50  using (($1)/100):2 with linespoints lt rgb "blue" lw 4 t "Baseline, same dataset","BASELINE_DIFF" every 50  using (($1)/100):2 with linespoints lt rgb "#A2692E" lw 4 t "Baseline, different dataset","USERNAME" every 50 using (($1)/100):2 with linespoints lt rgb "red" lw 4 t "No username anon., same dataset"

#,"USERNAME_DIFF" every 5 using (($1)/100):2 with linespoints lt rgb "#006400" lw 4 t "No username anon., different dataset"
