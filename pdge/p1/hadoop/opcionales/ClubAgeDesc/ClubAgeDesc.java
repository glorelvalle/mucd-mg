//package org.apache.hadoop.examples;
package uam;
import java.io.IOException;
import java.util.*;
import org.apache.commons.lang.math.NumberUtils;
import java.text.DecimalFormat;

import org.apache.hadoop.conf.*;
import org.apache.hadoop.mapreduce.*;
import org.apache.hadoop.io.*;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;


public class ClubAgeDesc {

  public static class ClubAgeDescMapper extends Mapper<Object, Text, Text, DoubleWritable> {

		private static final String SEPARATOR = ",";

		public void map(Object key, Text value, Context context) throws IOException, InterruptedException {
			final String[] values = value.toString().split(SEPARATOR);


			final String age = format(values[1]);
			final String club = format(values[2]);

			if (NumberUtils.isNumber(age.toString())) {
				context.write(new Text(club), new DoubleWritable(NumberUtils.toDouble(age)));
			}
		}

		private String format(String value) {
			return value.trim();
		}
	}
  
	public static class ClubAgeDescReducer extends Reducer<Text, DoubleWritable, Text, Text> {

		private final DecimalFormat decimalFormat = new DecimalFormat("#.##");

		public void reduce(Text key, Iterable<DoubleWritable> coValues, Context context) throws IOException, InterruptedException {
			int measures = 0;
			double totalCo = 0.0f;
			double min = Double.POSITIVE_INFINITY;
			double max = Double.NEGATIVE_INFINITY;

			for (DoubleWritable coValue : coValues) {
				double val = coValue.get();
				totalCo += val;
				measures++;

				if (min > val) {
					min = val;
				}
				if (max < val) {
					max = val;				
				}
			}
			

			if (measures > 0) {
				context.write(key, new Text(decimalFormat.format(totalCo / measures) + ' ' + decimalFormat.format(min) + ' ' + decimalFormat.format(max)));
			}
		}
	}

  public static void main(String[] args) throws Exception {
    Configuration conf = new Configuration();

    /*String[] otherArgs = new GenericOptionsParser(conf, args).getRemainingArgs();
    if (otherArgs.length != 2) {
      System.err.println("Usage: wordcount <in> <out>");
      System.exit(2);
    }*/

    @SuppressWarnings("deprecation")
    Job job = new Job(conf, "clubagedesc");
		job.setJarByClass(ClubAgeDesc.class);
		job.setInputFormatClass(TextInputFormat.class);
		job.setOutputFormatClass(TextOutputFormat.class);

		job.setMapperClass(ClubAgeDescMapper.class);
		job.setReducerClass(ClubAgeDescReducer.class);

		job.setMapOutputKeyClass(Text.class);
		job.setMapOutputValueClass(DoubleWritable.class);
		job.setOutputKeyClass(Text.class);
		job.setOutputValueClass(Text.class);

    FileInputFormat.addInputPath(job, new Path(args[0]));
    FileOutputFormat.setOutputPath(job, new Path(args[1]));

    System.exit(job.waitForCompletion(true) ? 0 : 1);
  }
}
