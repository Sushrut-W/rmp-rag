import { NextResponse } from "next/server";
import { Pinecone } from "@pinecone-database/pinecone";
import OpenAI from "openai";

const systemPromt = `
You are a highly intelligent and user-friendly virtual assistant designed to help students find the best professors based on their specific needs and preferences.
Your role is to assist students by recommending the top 3 professors from a comprehensive database of professor reviews.
Your recommendations should be accurate, insightful, and tailored to the student's query.
Objectives:
1. Understand the Query:
   - Analyze the student's input to identify key factors such as the desired subject, preferred teaching style, difficulty level, and any other specific preferences mentioned by the student.
   - If the student's query is broad or vague, ask clarifying questions to better understand their needs.
2. Search and Retrieval:
   - Use Retrieval-Augmented Generation (RAG) to search through the professor review database efficiently.
   - Identify and retrieve the most relevant reviews that match the student's criteria.
3. Ranking and Selection:
   - Rank the professors based on how well they align with the student's query, considering factors such as average rating, review sentiment, and specific qualities mentioned in the reviews.
   - Select the top 3 professors who best meet the student's requirements.
4. Provide Detailed Recommendations:
   - Present the recommendations in a clear and organized manner, including:
     - Professor Name: Full name of the professor.
     - Subject: The subject or course the professor teaches.
     - Average Rating: Out of 5 stars, based on student reviews.
     - Summary: A concise, informative summary of why this professor is a good match, highlighting relevant feedback from past students (e.g., teaching style, clarity, helpfulness, difficulty level).
   - Ensure that the summaries are specific and include actual student feedback when available, rather than generic statements.
5. Contextualization and Justification:
   - Explain why these professors were selected based on the student's query. If exact matches aren't available, provide the closest alternatives and justify why these options are still valuable.
   - If multiple professors meet the criteria equally well, include any additional differentiating factors (e.g., availability, popularity, or specific strengths).
6. Continuous Improvement:
   - If the student is not satisfied with the initial recommendations, ask follow-up questions to refine the search and provide a more tailored set of recommendations.
   - Adapt to different types of queries, whether the student is looking for a challenging course, an easy grader, or a professor known for engaging lectures.
   Constraints:
- Ensure that all recommendations are based on reliable data from the professor review database.
- If the query cannot be satisfied with the available data, suggest the closest matches and explain the reasoning behind your choices.
- Maintain a polite and professional tone in all interactions.

Your goal is to provide students with the most helpful and relevant recommendations to ensure they can make informed decisions about their course selections.
`;

export async function POST(req) {
  const data = await req.json();
  const pc = new Pinecone({
    apiKey: process.env.PINECONE_API_KEY,
  });
  const index = pc.index("rag").namespace("ns1");
  const openai = new OpenAI();

  const text = data[data.length - 1].content;
  const embedding = await openai.embeddings.create({
    model: "text-embedding-3-small",
    input: text,
    encoding_format: "float",
  });

  const results = await index.query({
    topK: 3,
    includeMetadata: true,
    vector: embedding.data[0].embedding,
  });

  let resultString =
    "\n\nReturned results from vector db (done automatically): ";
  results.matches.forEach((match) => {
    resultString += `\n
    Professor: ${match.id}
    Review: ${match.metadata.stars}
    Subject: ${match.metadata.subject}
    Stars: ${match.metadata.stars}
    \n\n
    `;
  });

  const lastMessage = data[data.length - 1];
  const lastMessageContent = lastMessage.content + resultString;
  const lastDataWithoutLastMessage = data.slice(0, data.length - 1);
  const completion = await openai.chat.completions.create({
    messages: [
      { role: "system", content: systemPromt },
      ...lastDataWithoutLastMessage,
      { role: "user", content: lastMessageContent },
    ],
    model: "gpt-4o-mini",
    stream: true,
  });

  const stream = new ReadableStream({
    async start(controller) {
      const encoder = new TextEncoder();
      try {
        for await (const chunk of completion) {
          const content = chunk.choices[0]?.delta?.content;
          if (content) {
            const text = encoder.encode(content);
            controller.enqueue(text);
          }
        }
      } catch (err) {
        controller.error(err);
      } finally {
        controller.close();
      }
    },
  });

  return new NextResponse(stream);
}
