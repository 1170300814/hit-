package UV_init;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class init {

  static Map<String, String> artistMap = new HashMap<String, String>();// 存入变体ID和正确ID的映射
  static List<String> list = new ArrayList<String>();// 存入清洗后的信息，一起写入磁盘中减少IO次数

  public static void main(String[] args) throws IOException {
    // TODO Auto-generated method stub
    File artistsFile = new File("E:\\DATAdig-DATA\\artist_alias.txt");

    BufferedReader bufferedReader = new BufferedReader(new FileReader(artistsFile));
    String readString = "";
    while ((readString = bufferedReader.readLine()) != null) {



      String[] strings = readString.split("\t");

      // if (strings.length == 2 && !strings[0].equals(""))
      // System.out.print(strings[0] + " yy " + strings[1] + "\n");


      artistMap.put(strings[0], strings[1]);
    }



    File file1 = new File("E:\\DATAdig-DATA\\artist_data.txt");

    BufferedReader bufferedReader2 = new BufferedReader(new FileReader(file1));

    String string = "";
    int count = 0;
    while ((string = bufferedReader2.readLine()) != null) {
      String[] strings = string.split("\t");


      // System.out.print(strings.length + "\n");


      if (strings.length == 2 && artistMap.containsKey(strings[0])) {
        count++;
        String splic = artistMap.get(strings[0]) + "\t" + strings[1];
        string = splic;
      } else if (strings.length == 2) {
        string = strings[0] + "\t" + strings[1];
      }


      if (strings.length == 2) {
        list.add(string);
      }


    }
    FileWriter fileWriter = new FileWriter(new File("E:\\DATAdig-DATA\\changed_artist_data.txt"));


    for (int i = 0; i < list.size(); i++) {
      fileWriter.write(list.get(i) + "\n");
    }


    System.out.println("count is " + count);

  }

}
