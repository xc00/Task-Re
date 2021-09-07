import com.fasterxml.jackson.databind.ObjectMapper;
import edu.seu.gavin.crawler.Crawler;
import edu.seu.gavin.crawler.domain.Developer;
import edu.seu.gavin.crawler.domain.Submitter;
import edu.seu.gavin.crawler.domain.Task;
import edu.seu.gavin.crawler.service.AuthorizationService;
import edu.seu.gavin.crawler.service.DeveloperService;
import edu.seu.gavin.crawler.service.TaskService;
import edu.seu.gavin.crawler.service.impl.AuthorizationServiceImpl;
import edu.seu.gavin.crawler.service.impl.DeveloperServiceImpl;
import edu.seu.gavin.crawler.service.impl.TaskServiceImpl;
import edu.seu.gavin.crawler.util.DateUtil;
import edu.seu.gavin.crawler.util.FileUtil;
import edu.seu.gavin.crawler.util.JsonFormatUtil;
import org.junit.Test;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.util.ArrayList;

public class MyTest {

    @Test
    public void testGetTopDevelopersByPage() {
        DeveloperService developerService = new DeveloperServiceImpl();
        ArrayList<Developer> developers = developerService.getTopDevelopersByPage(1, 100);
        System.out.println(developers);
    }

    @Test
    public void testGetHistoryTasksByHandle() {
        TaskService taskService = new TaskServiceImpl();
        ArrayList<Task> tasks = taskService.getHistoryTasksByHandle("saarixx", 0);
        System.out.println(tasks.size());
        System.out.println(tasks);
    }

    @Test
    public void testGetTaskDetail() {
        TaskService taskService = new TaskServiceImpl();
        Task task = taskService.getTaskDetail(24477821);
        System.out.println(task);
    }

    @Test
    public void testFromStringToDate() {
        System.out.println(DateUtil.fromStringToDate(null));
    }


    @Test
    public void testStringToJsonFile() {
        ObjectMapper mapper = new ObjectMapper();
        String json ;
        Developer developer = new Developer();
        developer.setId(111);
        developer.setHandle("gavinyan");
        developer.setHistoryTasks(new ArrayList<>());

        try {
            json = mapper.writeValueAsString(developer);
            json = JsonFormatUtil.formatJson(json);
            FileWriter fw = new FileWriter(new File(System.getProperty("user.dir") + "\\src\\main\\resources\\json\\" + "gavinyan" + ".json"));
            BufferedWriter bw = new BufferedWriter(fw);
            bw.write(json);
            bw.flush();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    @Test
    public void testGetAllTaskTechnologies() {

        Crawler crawler = new Crawler();
        crawler.getAllTaskTechnologies();
    }

    @Test
    public void testGetAllTaskTypes() {

        Crawler crawler = new Crawler();
        crawler.getAllTaskTypes();
    }

    @Test
    public void testGetAllTaskSubTypes() {

        Crawler crawler = new Crawler();
        crawler.getAllTaskSubTypes();
    }

    @Test
    public void testGetAllTaskStatus() {

        Crawler crawler = new Crawler();
        crawler.getAllTaskStatus();
    }

    @Test
    public void testFilterTasks() {

        Crawler crawler = new Crawler();
        crawler.filterInvalidTasks();
    }

    @Test
    public void testGetTaskResult(){

        TaskService taskService = new TaskServiceImpl();
        ArrayList<Submitter> submitters = taskService.getTaskResult(30101930);
        FileUtil.fromObjectToFile(FileUtil.ROOT_PATH + "submitter.json", submitters, false);

    }

}
