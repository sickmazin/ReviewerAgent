import { createBrowserRouter } from "react-router";
import HomePage from "./components/HomePage";
import ReviewPage from "./components/ReviewPage";

export const router = createBrowserRouter([
  {
    path: "/",
    Component: HomePage,
  },
  {
    path: "/review",
    Component: ReviewPage,
  },
  {
    path: "/review/:chatId",
    Component: ReviewPage,
  },
]);